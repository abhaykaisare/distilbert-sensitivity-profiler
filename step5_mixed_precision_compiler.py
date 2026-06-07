"""
step5_mixed_precision_compiler.py
----------------------------------
Purpose: Use the sensitivity map from step3 to build a mixed-precision model.

The core idea:
  Blind INT8 quantization hits every layer uniformly.
  Your profiler proved that not all layers deserve equal treatment —
  some layers (KL > threshold) are sensitive and lose accuracy when
  quantized. Others (KL <= threshold) are resilient and can safely
  take INT8 with almost zero accuracy impact.

  The mixed-precision compiler applies this knowledge:
    Sensitive layers  → stay at FP32  (protected)
    Resilient layers  → quantized to INT8  (compressed)

  The result is benchmarked against both the FP32 baseline and a
  naive full-INT8 model, giving you a three-way tradeoff comparison.

Outputs:
  mixed_precision_model.pt   — saved model state dict
  benchmark_report.txt       — three-way comparison (UTF-8 safe)

Run: python step5_mixed_precision_compiler.py
Requires: sensitivity_results.json from step3
"""

import torch
import copy
import json
import os
import time
import tempfile
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased-finetuned-sst-2-english"
RESULTS_FILE  = "sensitivity_results.json"
SAVED_MODEL   = "mixed_precision_model.pt"
REPORT_FILE   = "benchmark_report.txt"
EVAL_SAMPLES  = 400
BATCH_SIZE    = 16
LATENCY_RUNS  = 3       # average over this many runs for stable timing
KL_THRESHOLD  = 1e-4    # same threshold used in step3 and step4


# ── Quantization Helpers ───────────────────────────────────────────────────────

def get_parent_and_child(model: nn.Module, layer_name: str):
    parts  = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_mixed_precision_model(base_model: nn.Module,
                                 results: dict,
                                 threshold: float) -> tuple:
    """
    Builds the mixed-precision model from the sensitivity scan results.

    Strategy:
      - Deep-copy the FP32 model ONCE.
      - Iterate through all scanned layers.
      - For each resilient layer (KL <= threshold): quantize in-place.
      - For each sensitive layer (KL > threshold): leave untouched.

    Why one deep-copy instead of 17 separate ones:
      Each call to our step2/step3 quantize_one_layer did a full deep copy.
      Here we copy once and apply all quantizations to the same copy,
      which is far more efficient for building the final model.

    Returns: (mixed_model, list_of_quantized_layers, list_of_protected_layers)
    """
    model_copy      = copy.deepcopy(base_model)
    quantized_layers = []
    protected_layers = []

    for layer_name, metrics in results.items():
        parent, child_name = get_parent_and_child(model_copy, layer_name)
        original_layer     = getattr(parent, child_name)

        if metrics['kl_divergence'] <= threshold:
            # Resilient — quantize this layer to INT8
            wrapper  = nn.Sequential(original_layer)
            q_wrapper= torch.quantization.quantize_dynamic(
                           wrapper, {nn.Linear}, dtype=torch.qint8)
            setattr(parent, child_name, q_wrapper[0])
            quantized_layers.append(layer_name)
        else:
            # Sensitive — leave at FP32
            protected_layers.append(layer_name)

    return model_copy, quantized_layers, protected_layers


def build_full_int8_model(base_model: nn.Module) -> nn.Module:
    """
    Quantizes EVERY Linear layer to INT8, no exceptions.
    This is what standard blind quantization does.
    Used here only as a comparison baseline.
    """
    model_copy = copy.deepcopy(base_model)
    return torch.quantization.quantize_dynamic(
        model_copy, {nn.Linear}, dtype=torch.qint8
    )


# ── Evaluation Helpers ─────────────────────────────────────────────────────────

def evaluate(model: nn.Module, tokenizer, texts: list, labels: list) -> float:
    """Returns accuracy (0.0-1.0) on the given dataset."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch  = texts[start : start + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            preds  = torch.argmax(model(**inputs).logits, dim=-1).tolist()
            all_preds.extend(preds)
    return sum(p == l for p, l in zip(all_preds, labels)) / len(labels)


def benchmark_latency(model: nn.Module, tokenizer, texts: list) -> float:
    """
    Measures average inference latency per sentence in milliseconds.
    Runs LATENCY_RUNS full passes and averages to reduce timing noise.
    """
    model.eval()
    times = []

    with torch.no_grad():
        for _ in range(LATENCY_RUNS):
            t0 = time.time()
            for start in range(0, len(texts), BATCH_SIZE):
                batch  = texts[start : start + BATCH_SIZE]
                inputs = tokenizer(batch, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt")
                _ = model(**inputs).logits
            times.append(time.time() - t0)

    avg_total_sec   = sum(times) / LATENCY_RUNS
    ms_per_sentence = (avg_total_sec / len(texts)) * 1000
    return ms_per_sentence


def get_model_size_mb(model: nn.Module) -> float:
    """
    Saves the model state dict to a temp file and checks its size.
    Dynamic quantization stores Linear weights as INT8, so quantized
    models produce smaller files.
    """
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        tmp_path = f.name
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    os.unlink(tmp_path)
    return size_mb


def count_int8_layers(model: nn.Module) -> int:
    """Counts Linear layers that have been quantized to INT8."""
    count = 0
    for _, module in model.named_modules():
        # Dynamic quantized linear has a different class name
        if 'quantized' in type(module).__module__:
            count += 1
    return count


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_and_write(lines: list, filepath: str):
    """Prints to console and writes to file (UTF-8, safe on Windows)."""
    text = '\n'.join(lines)
    print(text)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  STEP 5: Mixed Precision Compiler")
    print("=" * 65)

    # ── Load sensitivity results ───────────────────────────────────────────────
    if not os.path.exists(RESULTS_FILE):
        print(f"\n  ERROR: {RESULTS_FILE} not found. Run step3 first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    total_layers    = len(results)
    resilient_count = sum(1 for r in results.values()
                          if r['kl_divergence'] <= KL_THRESHOLD)
    sensitive_count = total_layers - resilient_count

    print(f"\n  Loaded {total_layers} layer results.")
    print(f"  KL threshold     : {KL_THRESHOLD}")
    print(f"  Layers to quantize (resilient) : {resilient_count}")
    print(f"  Layers to protect (sensitive)  : {sensitive_count}")

    # ── Load model, tokenizer, dataset ────────────────────────────────────────
    print("\n  Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    fp32_model.eval()

    print(f"  Loading {EVAL_SAMPLES} SST-2 validation samples...")
    dataset = load_dataset("glue", "sst2", split=f"validation[:{EVAL_SAMPLES}]")
    texts   = dataset["sentence"]
    labels  = dataset["label"]

    # ── Build models ──────────────────────────────────────────────────────────
    print("\n  Building mixed-precision model...")
    mixed_model, q_layers, p_layers = build_mixed_precision_model(
        fp32_model, results, KL_THRESHOLD
    )
    print(f"    -> {len(q_layers)} layers quantized to INT8")
    print(f"    -> {len(p_layers)} layers kept at FP32")

    print("  Building full INT8 model (for comparison)...")
    int8_model = build_full_int8_model(fp32_model)

    # ── Save mixed precision model ─────────────────────────────────────────────
    torch.save(mixed_model.state_dict(), SAVED_MODEL)
    print(f"  Mixed precision model saved to: {SAVED_MODEL}")

    # ── Run all benchmarks ─────────────────────────────────────────────────────
    print("\n  Benchmarking — this will take a few minutes...")

    print("  [1/3] FP32 baseline...")
    acc_fp32      = evaluate(fp32_model, tokenizer, texts, labels)
    lat_fp32      = benchmark_latency(fp32_model, tokenizer, texts)
    size_fp32     = get_model_size_mb(fp32_model)

    print("  [2/3] Full INT8...")
    acc_int8      = evaluate(int8_model, tokenizer, texts, labels)
    lat_int8      = benchmark_latency(int8_model, tokenizer, texts)
    size_int8     = get_model_size_mb(int8_model)
    n_int8_layers = count_int8_layers(int8_model)

    print("  [3/3] Mixed precision...")
    acc_mixed     = evaluate(mixed_model, tokenizer, texts, labels)
    lat_mixed     = benchmark_latency(mixed_model, tokenizer, texts)
    size_mixed    = get_model_size_mb(mixed_model)
    n_mixed_layers= count_int8_layers(mixed_model)

    # ── Compute derived metrics ────────────────────────────────────────────────
    acc_drop_int8  = (acc_fp32 - acc_int8)  * 100
    acc_drop_mixed = (acc_fp32 - acc_mixed) * 100

    speedup_int8   = lat_fp32 / lat_int8
    speedup_mixed  = lat_fp32 / lat_mixed

    size_red_int8  = (1 - size_int8  / size_fp32) * 100
    size_red_mixed = (1 - size_mixed / size_fp32) * 100

    # How much of INT8's speedup did we capture?
    if speedup_int8 > 1.0:
        speedup_captured = ((speedup_mixed - 1) / (speedup_int8 - 1)) * 100
    else:
        speedup_captured = 0.0

    # ── Build report lines ─────────────────────────────────────────────────────
    lines = [
        "",
        "=" * 70,
        "  BENCHMARK REPORT — Three-Way Comparison",
        "=" * 70,
        "",
        f"  {'Model':<24} {'Accuracy':>10} {'Latency':>14} {'Size':>10} {'INT8 Layers':>14}",
        "  " + "-" * 66,
        f"  {'FP32 Baseline':<24} {acc_fp32*100:>9.2f}% {lat_fp32:>12.2f}ms {size_fp32:>8.1f}MB {0:>10}/{total_layers}",
        f"  {'Full INT8':<24} {acc_int8*100:>9.2f}% {lat_int8:>12.2f}ms {size_int8:>8.1f}MB {n_int8_layers:>10}/{total_layers}",
        f"  {'Mixed Precision':<24} {acc_mixed*100:>9.2f}% {lat_mixed:>12.2f}ms {size_mixed:>8.1f}MB {n_mixed_layers:>10}/{total_layers}",
        "  " + "-" * 66,
        "",
        "  ACCURACY IMPACT",
        f"    Full INT8 accuracy drop   : {acc_drop_int8:+.4f}%",
        f"    Mixed precision acc drop  : {acc_drop_mixed:+.4f}%",
        "",
        "  SPEED GAINED",
        f"    Full INT8 speedup         : {speedup_int8:.2f}x  (reference ceiling)",
        f"    Mixed precision speedup   : {speedup_mixed:.2f}x",
        f"    Speedup captured          : {speedup_captured:.1f}% of INT8's ceiling",
        "",
        "  SIZE REDUCTION (vs FP32)",
        f"    Full INT8 size reduction  : {size_red_int8:.1f}%",
        f"    Mixed precision reduction : {size_red_mixed:.1f}%",
        "",
        "  LAYER DECISIONS",
        f"    Quantized (INT8) : {len(q_layers)} layers  (resilient — KL <= {KL_THRESHOLD})",
        f"    Protected (FP32) : {len(p_layers)} layers  (sensitive — KL > {KL_THRESHOLD})",
        "",
        "  PROTECTED LAYERS (kept at FP32):",
    ]

    for name in sorted(p_layers, key=lambda n: results[n]['kl_divergence'], reverse=True):
        kl = results[name]['kl_divergence']
        lines.append(f"    KL={kl:.6f}  {name}")

    lines += [
        "",
        "  QUANTIZED LAYERS (converted to INT8):",
    ]
    for name in sorted(q_layers, key=lambda n: results[n]['kl_divergence'], reverse=True):
        kl = results[name]['kl_divergence']
        lines.append(f"    KL={kl:.6f}  {name}")

    lines += [
        "",
        "=" * 70,
        "  Next: step6_final_demo.py  — run inference on custom text",
        "=" * 70,
        "",
    ]

    print_and_write(lines, REPORT_FILE)
    print(f"\n  Report saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
