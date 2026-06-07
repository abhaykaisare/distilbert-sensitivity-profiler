"""
step2_single_layer_quant.py  (v2 — fixed metric)
-------------------------------------------------
What changed from v1:
  Hard accuracy was the wrong metric. INT8 quantization shifts logits
  slightly but rarely flips predictions on a well-trained 2-class model.
  The result: accuracy stayed frozen at 0.91 no matter which layer you
  touched, even though the model output WAS changing.

New primary metric: KL Divergence between original and quantized softmax
distributions. This is a continuous, information-theoretic measure of how
much a layer's quantization disturbed the model's confidence. Even tiny
logit shifts register as a non-zero KL score.

  KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
  P = softmax(original logits)
  Q = softmax(quantized logits)
  KL = 0.0 means quantization changed nothing.
  KL > 0.0 means the distribution shifted. Higher = more sensitive layer.

Hard accuracy is still reported as a secondary number, but it's not
the signal you're building the sensitivity map on.

Run: python step2_single_layer_quant.py
     Change TARGET_LAYER to different names from layer_names.txt.
     You will now see different KL scores per layer.
"""

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased-finetuned-sst-2-english"
EVAL_SAMPLES = 400   # more samples = more reliable signal
BATCH_SIZE   = 16
TARGET_LAYER = "distilbert.transformer.layer.5.attention.out_lin"


# ── Core Functions ─────────────────────────────────────────────────────────────

def get_parent_and_child(model: nn.Module, layer_name: str):
    """
    Traverses the module tree using the dot-separated layer name.
    Returns (parent_module, child_attribute_name).
    """
    parts  = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def quantize_one_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Returns a deep copy of the model with ONLY the named layer quantized
    to INT8. All other layers stay FP32. Original model is untouched.
    """
    model_copy = copy.deepcopy(model)
    parent, child_name = get_parent_and_child(model_copy, layer_name)

    original_layer    = getattr(parent, child_name)
    wrapper           = nn.Sequential(original_layer)
    quantized_wrapper = torch.quantization.quantize_dynamic(
        wrapper, {nn.Linear}, dtype=torch.qint8
    )
    setattr(parent, child_name, quantized_wrapper[0])
    return model_copy


def get_all_logits(model: nn.Module, tokenizer, texts: list) -> torch.Tensor:
    """
    Runs inference on all texts. Returns a [N, num_classes] tensor of raw
    logits — one row per sample, one column per class.

    We collect raw logits (not softmax) here and apply softmax later
    so the same logits can be reused for both KL and accuracy calculations.
    """
    model.eval()
    all_logits = []

    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start : start + BATCH_SIZE]
            inputs      = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            logits = model(**inputs).logits   # shape: [batch_size, 2]
            all_logits.append(logits)

    return torch.cat(all_logits, dim=0)       # shape: [N, 2]


def compute_kl_divergence(logits_orig: torch.Tensor, logits_quant: torch.Tensor) -> float:
    """
    Computes mean KL divergence between the original and quantized model's
    output probability distributions across all samples.

    KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
      P = softmax(original logits)  — treated as the "true" distribution
      Q = softmax(quantized logits) — the approximation we're measuring

    F.kl_div expects log-probabilities as the first argument and
    probabilities as the second. reduction='batchmean' averages over
    the batch dimension.

    Units: nats. Higher value = quantizing this layer disturbed the
    model's output distribution more = more sensitive layer.
    """
    P = F.softmax(logits_orig,  dim=-1)            # original probs
    Q = F.softmax(logits_quant, dim=-1)            # quantized probs

    kl = F.kl_div(Q.log(), P, reduction='batchmean')
    return kl.item()


def compute_accuracy(logits: torch.Tensor, labels: list) -> float:
    """Computes hard classification accuracy from logits."""
    preds   = torch.argmax(logits, dim=-1).tolist()
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels)


def compute_mean_logit_mse(logits_orig: torch.Tensor, logits_quant: torch.Tensor) -> float:
    """
    Mean squared error between original and quantized logits.
    A raw measure of how much the numbers changed.
    """
    return F.mse_loss(logits_quant, logits_orig).item()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  STEP 2 v2: Single Layer Sensitivity (KL Divergence)")
    print("=" * 65)

    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"Loading {EVAL_SAMPLES} SST-2 validation samples...")
    dataset = load_dataset("glue", "sst2", split=f"validation[:{EVAL_SAMPLES}]")
    texts   = dataset["sentence"]
    labels  = dataset["label"]

    # ── Collect original logits once (no need to rerun FP32 model later) ──────
    print("\n[1/3] Collecting original FP32 logits...")
    logits_fp32 = get_all_logits(model, tokenizer, texts)

    # ── Quantize target layer and collect its logits ───────────────────────────
    print(f"[2/3] Quantizing: {TARGET_LAYER}")
    quantized_model  = quantize_one_layer(model, TARGET_LAYER)
    logits_quantized = get_all_logits(quantized_model, tokenizer, texts)

    # ── Compute all metrics ───────────────────────────────────────────────────
    print("[3/3] Computing sensitivity metrics...\n")

    kl_score    = compute_kl_divergence(logits_fp32, logits_quantized)
    mse_score   = compute_mean_logit_mse(logits_fp32, logits_quantized)
    acc_fp32    = compute_accuracy(logits_fp32,      labels)
    acc_quant   = compute_accuracy(logits_quantized, labels)
    acc_drop    = (acc_fp32 - acc_quant) * 100

    # ── Verdict ───────────────────────────────────────────────────────────────
    # Threshold chosen empirically — tune after running several layers.
    # KL > 1e-4 is a reasonable starting point for "sensitive" on SST-2.
    verdict = "SENSITIVE  — keep at FP32" if kl_score > 1e-4 \
              else "RESILIENT  — safe to quantize"

    print("─" * 65)
    print(f"  Layer               : {TARGET_LAYER}")
    print(f"")
    print(f"  KL Divergence       : {kl_score:.6f}  ← PRIMARY sensitivity signal")
    print(f"  Logit MSE           : {mse_score:.6f}  ← how much raw numbers shifted")
    print(f"  Accuracy (FP32)     : {acc_fp32 * 100:.2f}%")
    print(f"  Accuracy (quantized): {acc_quant * 100:.2f}%")
    print(f"  Accuracy drop       : {acc_drop:+.4f}%  ← too coarse, often stays 0")
    print(f"")
    print(f"  Verdict             : {verdict}")
    print("─" * 65)

    print("\n── Expected behaviour ─────────────────────────────────────────────")
    print("  Run this script 4-5 times with different TARGET_LAYER values.")
    print("  KL scores WILL differ across layers now, even when accuracy drop")
    print("  stays at 0.0. That spread is exactly what step3 maps for all 52")
    print("  layers automatically.\n")

    print("── Layers to try next ─────────────────────────────────────────────")
    print("  distilbert.transformer.layer.5.ffn.lin2   (deep FFN — usually high KL)")
    print("  distilbert.transformer.layer.0.ffn.lin1   (early FFN)")
    print("  pre_classifier                             (just before output)")
    print("  classifier                                 (output head)\n")


if __name__ == "__main__":
    main()