"""
step6_final_demo.py  (fixed)
-----------------------------
Bug fix from v1:
  load_state_dict() failed because a quantized model's state dict uses
  different key names (_packed_params, scale, zero_point) than a standard
  FP32 model (weight, bias). They are incompatible structures.

  Fix: instead of saving/loading state dict, we rebuild the mixed-precision
  model architecture from scratch using sensitivity_results.json — the same
  way step5 built it. This takes a few seconds and avoids the format mismatch.

Run:
  python step6_final_demo.py
  python step6_final_demo.py "This product completely exceeded my expectations."

Requires: sensitivity_results.json from step3
"""

import sys
import copy
import time
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantLinear
except ImportError:
    from torch.nn.quantized.dynamic import Linear as DynamicQuantLinear

MODEL_NAME   = "distilbert-base-uncased-finetuned-sst-2-english"
RESULTS_FILE = "sensitivity_results.json"
KL_THRESHOLD = 1e-4

TEST_SENTENCES = [
    "The customer support team resolved my issue within minutes.",
    "I've never tasted a better cup of coffee in my entire life.",
    "This framework is elegant, fast, and incredibly well documented.",
    "The package arrived three weeks late and was completely damaged.",
    "I cannot believe how misleading this advertisement was.",
    "Worst experience I have ever had with any software product.",
    "It works, I suppose, but I expected significantly more for the price.",
    "The film was not without its charm, despite the predictable ending.",
    "Nobody complained, which I guess counts as a success.",
    "I wouldn't say it's bad, but I wouldn't recommend it either.",
    "The hardware is excellent but the software lets the whole thing down.",
    "Beautiful design, painfully slow performance.",
]

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}


def get_parent_and_child(model, layer_name):
    parts  = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_mixed_precision_model(base_model, results, threshold):
    model_copy  = copy.deepcopy(base_model)
    n_quantized = 0
    for layer_name, metrics in results.items():
        if metrics['kl_divergence'] <= threshold:
            parent, child_name = get_parent_and_child(model_copy, layer_name)
            original  = getattr(parent, child_name)
            wrapper   = nn.Sequential(original)
            q_wrapper = torch.quantization.quantize_dynamic(
                            wrapper, {nn.Linear}, dtype=torch.qint8)
            setattr(parent, child_name, q_wrapper[0])
            n_quantized += 1
    return model_copy, n_quantized


def count_int8_layers(model):
    return sum(1 for _, m in model.named_modules()
               if isinstance(m, DynamicQuantLinear))


def predict(model, tokenizer, text):
    model.eval()
    with torch.no_grad():
        inputs  = tokenizer(text, return_tensors="pt",
                            truncation=True, max_length=128)
        t0      = time.perf_counter()
        logits  = model(**inputs).logits
        latency = (time.perf_counter() - t0) * 1000
    probs      = F.softmax(logits, dim=-1).squeeze()
    label_id   = int(torch.argmax(probs).item())
    confidence = float(probs[label_id].item())
    return label_id, confidence, latency


def bar(confidence, width=20):
    filled = round(confidence * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found. Run step3 first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    total_layers = len(results)

    print("\nLoading FP32 model...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    fp32_model.eval()

    print("Building mixed-precision model from sensitivity_results.json...")
    mixed_model, n_quantized = build_mixed_precision_model(
        fp32_model, results, KL_THRESHOLD)
    mixed_model.eval()
    n_int8 = count_int8_layers(mixed_model)
    print(f"  -> {n_int8} layers quantized to INT8, "
          f"{total_layers - n_int8} layers protected at FP32\n")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"{'='*62}")
        print(f"  INPUT: \"{text}\"")
        print(f"{'='*62}")
        fp_label, fp_conf, fp_lat = predict(fp32_model,  tokenizer, text)
        mx_label, mx_conf, mx_lat = predict(mixed_model, tokenizer, text)
        match = "MATCH" if fp_label == mx_label else "DIFFER"
        print(f"\n  FP32 Model      : {LABELS[fp_label]:<10} {fp_conf*100:.1f}%  "
              f"{bar(fp_conf)}  {fp_lat:.1f}ms")
        print(f"  Mixed Precision : {LABELS[mx_label]:<10} {mx_conf*100:.1f}%  "
              f"{bar(mx_conf)}  {mx_lat:.1f}ms")
        print(f"  Prediction      : {match}\n")
        return

    print(f"{'='*70}")
    print(f"  SIDE-BY-SIDE: FP32 vs Mixed Precision ({n_int8}/38 layers INT8)")
    print(f"{'='*70}\n")

    fp32_times, mixed_times = [], []
    mismatches = 0

    for i, sentence in enumerate(TEST_SENTENCES, 1):
        fp_label, fp_conf, fp_lat = predict(fp32_model,  tokenizer, sentence)
        mx_label, mx_conf, mx_lat = predict(mixed_model, tokenizer, sentence)
        fp32_times.append(fp_lat)
        mixed_times.append(mx_lat)
        if fp_label != mx_label:
            mismatches += 1
        tag   = "OK" if fp_label == mx_label else "!!"
        short = sentence[:54] + "..." if len(sentence) > 54 else sentence
        print(f"  {i:>2}. {short}")
        print(f"      FP32  : {LABELS[fp_label]:<9} {fp_conf*100:.1f}%  {bar(fp_conf, 18)}")
        print(f"      Mixed : {LABELS[mx_label]:<9} {mx_conf*100:.1f}%  {bar(mx_conf, 18)}  [{tag}]")
        print()

    avg_fp32  = sum(fp32_times)  / len(fp32_times)
    avg_mixed = sum(mixed_times) / len(mixed_times)
    speedup   = avg_fp32 / avg_mixed if avg_mixed > 0 else 1.0

    print(f"{'='*70}")
    print(f"  Sentences tested      : {len(TEST_SENTENCES)}")
    print(f"  Prediction mismatches : {mismatches}")
    print(f"  Avg latency  FP32     : {avg_fp32:.1f} ms/sentence")
    print(f"  Avg latency  Mixed    : {avg_mixed:.1f} ms/sentence")
    print(f"  Speedup               : {speedup:.2f}x")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()