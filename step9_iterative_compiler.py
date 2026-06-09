"""
step9_iterative_compiler.py  (v2 — with empirical validation)
--------------------------------------------------------------
v1 problem:
  The pairwise interaction matrix allowed many layers to have negative
  adjusted scores (synergies), which pulled the running KL total DOWN.
  The compiler quantized 36/38 layers and estimated total KL = 0.009447
  — well under budget — but real accuracy dropped 1.25%.

  Root cause: the 2D interaction matrix is a pairwise (second-order)
  approximation. When 36 layers are quantized simultaneously, higher-
  order effects (3-way, 4-way interactions) dominate and the pairwise
  estimate becomes dangerously optimistic.

v2 fix — empirical validation every N steps:
  Every VALIDATION_INTERVAL layers committed, build the current partial
  model and measure its REAL KL against the FP32 baseline. If real KL
  exceeds REAL_KL_LIMIT, stop — regardless of what the matrix predicts.

  This turns the greedy algorithm from "trust the matrix completely"
  into "use the matrix to guide decisions, reality to limit them."

Two stopping conditions now:
  1. Adjusted budget exhausted  (matrix-estimated cost exceeds KL_BUDGET)
  2. Real KL limit hit          (actual measured model KL exceeds REAL_KL_LIMIT)

REAL_KL_LIMIT is calibrated to give 0% accuracy drop on SST-2. From
step3 results, the 17 resilient layers collectively cause ~0.001 total
KL. Staying under 0.005 should preserve accuracy while allowing the
greedy compiler to find better combinations than step5's binary cut.
"""

import torch
import copy
import json
import os
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_NAME          = "distilbert-base-uncased-finetuned-sst-2-english"
RESULTS_FILE        = "sensitivity_results.json"
LAYER_FILE          = "docs/layer_names.txt"
MATRIX_FILE         = "interaction_matrix.npy"
PAIRWISE_FILE       = "pairwise_kl_scores.json"
REPORT_FILE         = "greedy_compiler_report.txt"

EVAL_SAMPLES        = 400
BATCH_SIZE          = 16
LATENCY_RUNS        = 3
KL_BUDGET           = 0.015    # matrix-estimated budget ceiling
REAL_KL_LIMIT       = 0.005    # real measured KL ceiling — the hard stop
VALIDATION_INTERVAL = 5        # validate every N committed layers
STEP5_THRESHOLD     = 1e-4


# ── Matrix / Model Helpers ─────────────────────────────────────────────────────

def load_or_rebuild_matrix(layer_names, isolated_kl):
    if os.path.exists(MATRIX_FILE):
        print(f"  Loading {MATRIX_FILE}...")
        return np.load(MATRIX_FILE)
    if not os.path.exists(PAIRWISE_FILE):
        raise FileNotFoundError(f"Need {MATRIX_FILE} or {PAIRWISE_FILE}. Run step7 first.")
    print(f"  Rebuilding matrix from {PAIRWISE_FILE}...")
    with open(PAIRWISE_FILE, 'r') as f:
        pairwise = json.load(f)
    n      = len(layer_names)
    matrix = np.zeros((n, n))
    for key, ckl in pairwise.items():
        i, j   = map(int, key.split(','))
        ni, nj = layer_names[i], layer_names[j]
        val    = ckl - isolated_kl.get(ni, 0) - isolated_kl.get(nj, 0)
        matrix[i][j] = val
        matrix[j][i] = val
    np.save(MATRIX_FILE, matrix)
    return matrix


def get_parent_and_child(model, layer_name):
    parts  = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def build_model_from_indices(base_model, layer_names, indices):
    """One deep copy, quantize only the given layer indices."""
    model_copy = copy.deepcopy(base_model)
    for idx in indices:
        name          = layer_names[idx]
        parent, child = get_parent_and_child(model_copy, name)
        orig          = getattr(parent, child)
        wrapper       = nn.Sequential(orig)
        q_wrap        = torch.quantization.quantize_dynamic(
                            wrapper, {nn.Linear}, dtype=torch.qint8)
        setattr(parent, child, q_wrap[0])
    return model_copy


# ── Inference / Metrics ────────────────────────────────────────────────────────

def get_all_logits(model, tokenizer, texts):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch  = texts[start : start + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            out.append(model(**inputs).logits)
    return torch.cat(out, dim=0)


def compute_kl(logits_orig, logits_quant):
    P = F.softmax(logits_orig,  dim=-1)
    Q = F.softmax(logits_quant, dim=-1)
    return F.kl_div(Q.log(), P, reduction='batchmean').item()


def compute_accuracy(logits, labels):
    preds   = torch.argmax(logits, dim=-1).tolist()
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def benchmark_latency(model, tokenizer, texts):
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
    return (sum(times) / LATENCY_RUNS / len(texts)) * 1000


def shorten(name):
    if name == 'pre_classifier': return 'pre_cls'
    if name == 'classifier':     return 'cls'
    parts = name.split('.')
    lnum  = parts[3]; ltype = parts[4]; sub = parts[5]
    if ltype == 'attention':
        m = {'q_lin':'q','k_lin':'k','v_lin':'v','out_lin':'out'}
        return f"L{lnum}.attn.{m.get(sub,sub)}"
    return f"L{lnum}.ffn.{sub.replace('lin','')}"


# ── Greedy Compiler v2 ─────────────────────────────────────────────────────────

def run_greedy_compiler_v2(layer_names, isolated_kl, matrix,
                            kl_budget, real_kl_limit, validation_interval,
                            base_model, tokenizer, texts, logits_fp32):
    """
    Greedy compiler with empirical validation.

    Uses the interaction matrix to guide layer selection (cheap decisions),
    but periodically validates against the real partially-quantized model
    (expensive but accurate). Stops when either:
      - Matrix-estimated accumulated cost > kl_budget
      - Real measured KL of current model > real_kl_limit

    Returns: (quantized_indices, trace, stop_reason)
    """
    n = len(layer_names)
    adjusted    = {i: isolated_kl.get(layer_names[i], 0.0) for i in range(n)}
    remaining   = list(range(n))
    quantized   = []
    accumulated = 0.0
    trace       = []
    last_real_kl = 0.0

    while remaining:
        # ── Select cheapest remaining layer ───────────────────────────────────
        best       = min(remaining, key=lambda i: adjusted[i])
        best_score = adjusted[best]

        # Stop 1: matrix budget exhausted
        if accumulated + best_score > kl_budget:
            trace.append({'action':'STOP_BUDGET', 'layer_name': layer_names[best],
                          'accumulated': accumulated, 'best_score': best_score})
            return quantized, trace, "MATRIX_BUDGET_EXHAUSTED"

        # Commit
        quantized.append(best)
        remaining.remove(best)
        iso = isolated_kl.get(layer_names[best], 0)
        accumulated += best_score

        trace.append({
            'step'        : len(quantized),
            'action'      : 'COMMIT',
            'layer_idx'   : best,
            'layer_name'  : layer_names[best],
            'isolated_kl' : iso,
            'adjusted_kl' : best_score,
            'accumulated' : accumulated,
            'real_kl'     : None,    # filled in at validation steps
        })

        # Update adjusted scores via interaction matrix
        for j in remaining:
            adjusted[j] += matrix[best][j]

        # ── Periodic empirical validation ─────────────────────────────────────
        if len(quantized) % validation_interval == 0:
            print(f"    [validate @ step {len(quantized)}] "
                  f"building partial model to measure real KL...")
            current_model = build_model_from_indices(
                base_model, layer_names, quantized)
            logits_current = get_all_logits(current_model, tokenizer, texts)
            real_kl        = compute_kl(logits_fp32, logits_current)
            last_real_kl   = real_kl
            trace[-1]['real_kl'] = real_kl

            print(f"    Real KL = {real_kl:.6f}  "
                  f"(limit = {real_kl_limit}, estimated = {accumulated:.6f})")

            # Stop 2: real model is already too degraded
            if real_kl > real_kl_limit:
                trace.append({'action': 'STOP_REAL_KL',
                               'real_kl': real_kl,
                               'accumulated': accumulated})
                # Roll back the last validation_interval layers to find the
                # largest safe set — binary search between (current-N, current)
                # For simplicity, roll back to the previous checkpoint
                safe_count = len(quantized) - validation_interval
                quantized  = quantized[:safe_count]
                return quantized, trace, f"REAL_KL_LIMIT (measured {real_kl:.6f} > {real_kl_limit})"

    return quantized, trace, "ALL_LAYERS_PROCESSED"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  STEP 9 v2: Iterative Greedy Compiler (with Empirical Validation)")
    print("=" * 70)

    for fname in [RESULTS_FILE, LAYER_FILE]:
        if not os.path.exists(fname):
            print(f"\n  ERROR: {fname} not found."); return

    with open(RESULTS_FILE, 'r') as f: step3 = json.load(f)
    with open(LAYER_FILE,   'r') as f:
        layer_names = [l.strip() for l in f if l.strip()]

    isolated_kl = {name: step3[name]['kl_divergence']
                   for name in layer_names if name in step3}
    n = len(layer_names)

    step5_set = {i for i, name in enumerate(layer_names)
                 if isolated_kl.get(name, 0) <= STEP5_THRESHOLD}

    print()
    matrix = load_or_rebuild_matrix(layer_names, isolated_kl)

    print("\n  Loading model and dataset...")
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    fp32_model.eval()

    dataset = load_dataset("glue", "sst2", split=f"validation[:{EVAL_SAMPLES}]")
    texts   = dataset["sentence"]
    labels  = dataset["label"]

    print("  Computing FP32 baseline logits...")
    logits_fp32 = get_all_logits(fp32_model, tokenizer, texts)

    # ── Run greedy compiler v2 ─────────────────────────────────────────────────
    print(f"\n  Running greedy compiler v2...")
    print(f"  Budget = {KL_BUDGET} (matrix)   "
          f"Real KL limit = {REAL_KL_LIMIT}   "
          f"Validation every {VALIDATION_INTERVAL} layers\n")

    quantized, trace, stop_reason = run_greedy_compiler_v2(
        layer_names, isolated_kl, matrix,
        KL_BUDGET, REAL_KL_LIMIT, VALIDATION_INTERVAL,
        fp32_model, tokenizer, texts, logits_fp32
    )
    greedy_set   = set(quantized)
    newly_quant  = greedy_set - step5_set
    newly_prot   = step5_set  - greedy_set

    # ── Print decision trace ───────────────────────────────────────────────────
    print(f"\n  Decision trace:\n")
    print(f"  {'Step':<5} {'Layer':<22} {'Isolated':>12} {'Adjusted':>12} "
          f"{'Accum':>12}  {'Real KL':>10}")
    print("  " + "─" * 82)

    for entry in trace:
        if entry['action'] == 'COMMIT':
            real_str = f"{entry['real_kl']:.6f}" if entry['real_kl'] else "       -"
            iso_adj_diff = entry['adjusted_kl'] - entry['isolated_kl']
            flag = " <NEW" if entry['layer_name'] in [layer_names[i] for i in newly_quant] else ""
            print(f"  {entry['step']:<5} {shorten(entry['layer_name']):<22} "
                  f"{entry['isolated_kl']:>12.6f} {entry['adjusted_kl']:>12.6f} "
                  f"{entry['accumulated']:>12.6f}  {real_str:>10}{flag}")
        elif 'STOP' in entry['action']:
            print(f"\n  STOPPED: {stop_reason}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    print(f"\n  Building models for benchmark...")

    print(f"  Step5 ({len(step5_set)} INT8 layers)...")
    step5_model = build_model_from_indices(fp32_model, layer_names, list(step5_set))

    print(f"  Greedy v2 ({len(greedy_set)} INT8 layers)...")
    greedy_model = build_model_from_indices(fp32_model, layer_names, quantized)

    print("  Benchmarking...")
    acc_fp32 = compute_accuracy(logits_fp32, labels)
    lat_fp32 = benchmark_latency(fp32_model, tokenizer, texts)

    logits_s5 = get_all_logits(step5_model,  tokenizer, texts)
    acc_s5    = compute_accuracy(logits_s5, labels)
    lat_s5    = benchmark_latency(step5_model, tokenizer, texts)

    logits_gr = get_all_logits(greedy_model, tokenizer, texts)
    acc_gr    = compute_accuracy(logits_gr, labels)
    lat_gr    = benchmark_latency(greedy_model, tokenizer, texts)

    drop_s5 = (acc_fp32 - acc_s5) * 100
    drop_gr = (acc_fp32 - acc_gr) * 100
    spd_s5  = lat_fp32 / lat_s5  if lat_s5  > 0 else 1.0
    spd_gr  = lat_fp32 / lat_gr  if lat_gr  > 0 else 1.0

    # ── Build report ──────────────────────────────────────────────────────────
    lines = [
        "", "=" * 70,
        "  STEP 9 v2: GREEDY COMPILER — FINAL REPORT",
        "=" * 70, "",
        "  SETTINGS",
        f"    KL Budget (matrix)  : {KL_BUDGET}",
        f"    Real KL limit       : {REAL_KL_LIMIT}",
        f"    Validation interval : every {VALIDATION_INTERVAL} layers",
        f"    Stop reason         : {stop_reason}",
        "",
        "  THREE-WAY BENCHMARK",
        f"  {'Model':<22} {'Accuracy':>10} {'Acc Drop':>10} "
        f"{'Latency':>12} {'Speedup':>10} {'INT8 Layers':>13}",
        "  " + "-" * 80,
        f"  {'FP32 Baseline':<22} {acc_fp32*100:>9.2f}% {0:>10.4f}% "
        f"{lat_fp32:>10.2f}ms {'1.00x':>10} {0:>7}/{n}",
        f"  {'Step5 (binary)':<22} {acc_s5*100:>9.2f}% {drop_s5:>+10.4f}% "
        f"{lat_s5:>10.2f}ms {spd_s5:>9.2f}x {len(step5_set):>7}/{n}",
        f"  {'Step9-v2 (greedy)':<22} {acc_gr*100:>9.2f}% {drop_gr:>+10.4f}% "
        f"{lat_gr:>10.2f}ms {spd_gr:>9.2f}x {len(greedy_set):>7}/{n}",
        "  " + "-" * 80, "",
        "  LAYER DECISIONS",
        f"    Newly quantized by greedy ({len(newly_quant)} vs step5):",
    ]
    for idx in sorted(newly_quant):
        name = layer_names[idx]
        iso  = isolated_kl.get(name, 0)
        adj  = next((e['adjusted_kl'] for e in trace
                     if e.get('layer_idx') == idx
                     and e['action'] == 'COMMIT'), iso)
        lines.append(f"      {shorten(name):<22} isolated={iso:.6f}  "
                     f"adjusted={adj:.6f}  delta={adj-iso:+.6f}")

    lines += ["",
              f"    Newly protected by greedy ({len(newly_prot)} vs step5):"]
    for idx in sorted(newly_prot):
        lines.append(f"      {shorten(layer_names[idx])}")

    lines += ["", "=" * 70, ""]
    text = '\n'.join(lines)
    print(text)
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(f"  Report saved to: {REPORT_FILE}\n")
    with open("greedy_layer_decisions.json", 'w') as f:
        json.dump({'quantized_names': [layer_names[i] for i in quantized]}, f, indent=2)
    print(f"  Layer decisions saved to: greedy_layer_decisions.json")


if __name__ == "__main__":
    main()