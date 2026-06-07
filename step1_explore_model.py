"""
step1_explore_model.py
----------------------
Purpose: Map every quantizable layer inside DistilBERT before touching anything.

Why this matters:
  The full sensitivity scan in step3 will iterate over every layer in this list
  one by one. You need to see this map first so you understand WHAT you're
  probing and WHY certain layers are expected to be more sensitive than others.

Run: python step1_explore_model.py
"""

import torch
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


def main():
    print("=" * 75)
    print("  STEP 1: DistilBERT Layer Map")
    print("=" * 75)

    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded.\n")

    # ── Why only nn.Linear? ───────────────────────────────────────────────────
    # Dynamic INT8 quantization targets matrix multiplication ops only.
    # These live inside nn.Linear layers (y = xW^T + b).
    # Embedding layers, LayerNorm, GELU activations — all stay FP32 regardless.
    # So your profiler only needs to probe Linear layers.

    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]

    # ── Print the layer map ───────────────────────────────────────────────────
    col_w = 62
    header = f"{'#':<5}{'Layer Name':<{col_w}}{'Weight Shape':<18}{'Params':>10}"
    print(header)
    print("─" * len(header))

    for i, (name, module) in enumerate(linear_layers):
        shape_str   = str(tuple(module.weight.shape))
        param_count = module.weight.numel()
        print(f"{i:<5}{name:<{col_w}}{shape_str:<18}{param_count:>10,}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_params = sum(m.weight.numel() for _, m in linear_layers)
    print("─" * len(header))
    print(f"\nTotal Linear layers      : {len(linear_layers)}")
    print(f"Total params in them     : {total_params:,}")

    # ── Structural breakdown ──────────────────────────────────────────────────
    # Help you mentally group the layers before the scan begins.
    print("\n── Layer Groups ──────────────────────────────────────────────────────")

    attention_layers = [(n, m) for n, m in linear_layers if "attention" in n]
    ffn_layers       = [(n, m) for n, m in linear_layers if "ffn" in n or "lin" in n.split(".")[-1] and "attention" not in n]
    classifier_layers= [(n, m) for n, m in linear_layers if "classifier" in n or "pre_classifier" in n]

    print(f"  Attention layers  ({len(attention_layers)}): Q, K, V + output projections per transformer block")
    print(f"  FFN layers        ({len(ffn_layers) - len(attention_layers)}): Two feed-forward linear layers per block")
    print(f"  Classifier head   ({len(classifier_layers)}): Final layers that output the sentiment logits")

    print("\n── What to expect in step3 ───────────────────────────────────────────")
    print(f"  The scanner will run {len(linear_layers)} experiments total.")
    print(f"  Each experiment: quantize ONE layer, measure accuracy drop on SST-2.")
    print(f"  The result is a sensitivity score (accuracy drop) for every layer above.")

    # ── Save layer names to a file (used by step3) ────────────────────────────
    layer_names = [name for name, _ in linear_layers]
    with open("layer_names.txt", "w") as f:
        for name in layer_names:
            f.write(name + "\n")

    print(f"\nLayer names saved to layer_names.txt (step3 will read this file)")


if __name__ == "__main__":
    main()
