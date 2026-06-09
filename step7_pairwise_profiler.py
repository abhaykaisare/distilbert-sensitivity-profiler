"""
step7_pairwise_profiler.py
--------------------------
Purpose: Measure how every pair of layers interacts when quantized
simultaneously. This reveals where step5's independent assumption breaks down.

The core formula:
  Interaction(i, j) = KL(both i and j quantized) - KL(i alone) - KL(j alone)

  Positive  → super-additive: errors compound, worse than predicted
  Near zero → independent:    step5's assumption holds for this pair
  Negative  → sub-additive:   errors partially cancel, safer than predicted

Why this matters:
  Step5 made every layer's quantization decision as if all other layers
  were perfect FP32. That's only true when measuring isolated sensitivity.
  In the actual mixed-precision model, multiple layers are quantized at once.
  This script measures the real cost of quantizing pairs together.

Output is a 38x38 interaction matrix — the input to step9's iterative
compiler, which uses it to make smarter multi-layer decisions.

Runtime : ~3 hours on CPU (703 pairs x ~16 sec each).
          Start this before stepping away. Fully resumable.

Checkpoint: saves after EVERY pair. If interrupted, re-run and it
            picks up from where it stopped.

Outputs:
  pairwise_kl_scores.json      raw combined KL per pair (checkpoint file)
  interaction_matrix.npy       38x38 numpy matrix
  interaction_preview.txt      human-readable top toxic / synergistic pairs

Requires: sensitivity_results.json (step3), layer_names.txt (step1)
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
MODEL_NAME     = "distilbert-base-uncased-finetuned-sst-2-english"
RESULTS_FILE   = "sensitivity_results.json"
LAYER_FILE     = "docs/layer_names.txt"
PAIRWISE_FILE  = "pairwise_kl_scores.json"
MATRIX_FILE    = "interaction_matrix.npy"
PREVIEW_FILE   = "interaction_preview.txt"
EVAL_SAMPLES   = 400
BATCH_SIZE     = 16


# ── Core Functions ─────────────────────────────────────────────────────────────

def get_parent_and_child(model, layer_name):
    parts  = layer_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def quantize_two_layers(model, name_i, name_j):
    """
    Returns a deep copy of the model with BOTH named layers quantized to INT8.

    Design decision — one deep copy, two in-place quantizations:
      Calling quantize_one_layer twice would create two separate deep copies
      and discard the first. This function copies once, then applies both
      quantizations to the same copy. Saves one full model copy per experiment
      (~250MB avoided), which matters over 703 experiments.

    The order of quantization doesn't affect the result since each layer's
    weights are independent of the other's quantization step.
    """
    model_copy = copy.deepcopy(model)

    for layer_name in [name_i, name_j]:
        parent, child_name = get_parent_and_child(model_copy, layer_name)
        original  = getattr(parent, child_name)
        wrapper   = nn.Sequential(original)
        q_wrapper = torch.quantization.quantize_dynamic(
                        wrapper, {nn.Linear}, dtype=torch.qint8)
        setattr(parent, child_name, q_wrapper[0])

    return model_copy


def get_all_logits(model, tokenizer, texts):
    """Runs inference on all texts. Returns [N, num_classes] logit tensor."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch  = texts[start : start + BATCH_SIZE]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            all_logits.append(model(**inputs).logits)
    return torch.cat(all_logits, dim=0)


def compute_kl(logits_orig, logits_quant):
    """KL divergence between FP32 and quantized output distributions."""
    P = F.softmax(logits_orig,  dim=-1)
    Q = F.softmax(logits_quant, dim=-1)
    return F.kl_div(Q.log(), P, reduction='batchmean').item()


# ── Checkpoint Helpers ─────────────────────────────────────────────────────────

def load_pairwise_scores():
    """Load previously completed pair results from disk."""
    if os.path.exists(PAIRWISE_FILE):
        with open(PAIRWISE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_pairwise_scores(scores):
    """Persist current results immediately. Called after every single pair."""
    with open(PAIRWISE_FILE, 'w') as f:
        json.dump(scores, f, indent=2)


# ── Matrix Builder ─────────────────────────────────────────────────────────────

def build_interaction_matrix(layer_names, isolated_kl, pairwise_kl):
    """
    Constructs the full 38x38 interaction matrix from:
      - isolated_kl : KL scores from step3 (each layer alone)
      - pairwise_kl : combined KL scores from this script (both layers)

    interaction(i,j) = KL(both) - KL(i alone) - KL(j alone)

    The matrix is symmetric: interaction(i,j) == interaction(j,i).
    Diagonal is zero: a layer doesn't interact with itself.
    """
    n      = len(layer_names)
    matrix = np.zeros((n, n))

    for key, combined_kl in pairwise_kl.items():
        i, j   = map(int, key.split(','))
        name_i = layer_names[i]
        name_j = layer_names[j]
        kl_i   = isolated_kl.get(name_i, 0.0)
        kl_j   = isolated_kl.get(name_j, 0.0)
        val    = combined_kl - kl_i - kl_j
        matrix[i][j] = val
        matrix[j][i] = val   # symmetric

    return matrix


# ── Display Helpers ────────────────────────────────────────────────────────────

def shorten(name):
    """Compact display name: 'distilbert.transformer.layer.2.ffn.lin1' -> 'L2.ffn.1'"""
    if name == 'pre_classifier': return 'pre_cls'
    if name == 'classifier':     return 'cls'
    parts = name.split('.')
    lnum  = parts[3]
    ltype = parts[4]
    sub   = parts[5]
    if ltype == 'attention':
        m = {'q_lin': 'q', 'k_lin': 'k', 'v_lin': 'v', 'out_lin': 'out'}
        return f"L{lnum}.attn.{m.get(sub, sub)}"
    return f"L{lnum}.ffn.{sub.replace('lin', '')}"


def format_eta(seconds):
    if seconds < 60:   return f"{int(seconds)}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def classify(interaction):
    if   interaction >  1e-4: return "SUPER-ADD"
    elif interaction < -1e-4: return "SUB-ADD  "
    else:                     return "INDEPEND "


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  STEP 7: Pairwise Interaction Profiler")
    print("=" * 70)

    # ── Validate inputs ───────────────────────────────────────────────────────
    for fname in [RESULTS_FILE, LAYER_FILE]:
        if not os.path.exists(fname):
            print(f"\n  ERROR: {fname} not found.")
            print(f"  Run step3 (for {RESULTS_FILE}) and step1 (for {LAYER_FILE}) first.")
            return

    with open(RESULTS_FILE, 'r') as f:
        step3_results = json.load(f)

    with open(LAYER_FILE, 'r') as f:
        layer_names = [line.strip() for line in f if line.strip()]

    # Isolated KL scores from step3 — real measured values, not estimates
    isolated_kl = {
        name: step3_results[name]['kl_divergence']
        for name in layer_names
        if name in step3_results
    }

    n           = len(layer_names)
    total_pairs = n * (n - 1) // 2

    # ── Resume check ──────────────────────────────────────────────────────────
    pairwise_kl = load_pairwise_scores()
    completed   = set(pairwise_kl.keys())

    remaining = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if f"{i},{j}" not in completed
    ]

    print(f"\n  Layers            : {n}")
    print(f"  Total pairs       : {total_pairs}")
    print(f"  Already completed : {len(completed)}")
    print(f"  Remaining         : {len(remaining)}")
    print(f"  Estimated runtime : ~{len(remaining) * 16 / 3600:.1f} hours")

    if not remaining:
        print("\n  All pairs already profiled.")
        print(f"  Delete {PAIRWISE_FILE} to restart from scratch.")
        matrix = build_interaction_matrix(layer_names, isolated_kl, pairwise_kl)
        np.save(MATRIX_FILE, matrix)
        print(f"  Interaction matrix rebuilt and saved to {MATRIX_FILE}")
        return

    # ── Load model, tokenizer, dataset ────────────────────────────────────────
    print("\n  Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(f"  Loading {EVAL_SAMPLES} SST-2 validation samples...")
    dataset = load_dataset("glue", "sst2", split=f"validation[:{EVAL_SAMPLES}]")
    texts   = dataset["sentence"]

    # Baseline FP32 logits — computed once, reused across all 703 experiments
    print("\n  Collecting FP32 baseline logits (once, reused for all pairs)...")
    logits_fp32 = get_all_logits(model, tokenizer, texts)
    print("  Baseline ready.\n")

    # ── Table header ──────────────────────────────────────────────────────────
    W = 16
    print(f"  {'i,j':<8} {'Layer i':<{W}} {'Layer j':<{W}} "
          f"{'KL_i':>10} {'KL_j':>10} {'KL_both':>10} {'Interaction':>13}  Type       ETA")
    print("  " + "─" * 100)

    # ── Main pairwise loop ────────────────────────────────────────────────────
    scan_start = time.time()
    done_count = 0

    for i, j in remaining:
        name_i = layer_names[i]
        name_j = layer_names[j]

        # The core experiment: quantize both, measure combined distribution shift
        q_model      = quantize_two_layers(model, name_i, name_j)
        logits_quant = get_all_logits(q_model, tokenizer, texts)
        combined_kl  = compute_kl(logits_fp32, logits_quant)

        # Interaction = deviation from the independent assumption
        kl_i        = isolated_kl.get(name_i, 0.0)
        kl_j        = isolated_kl.get(name_j, 0.0)
        interaction = combined_kl - kl_i - kl_j

        # Checkpoint immediately — never lose a completed pair
        pairwise_kl[f"{i},{j}"] = combined_kl
        save_pairwise_scores(pairwise_kl)

        done_count += 1
        elapsed    = time.time() - scan_start
        eta        = (elapsed / done_count) * (len(remaining) - done_count)

        si = shorten(name_i)
        sj = shorten(name_j)
        print(f"  {i},{j:<6} {si:<{W}} {sj:<{W}} "
              f"{kl_i:>10.6f} {kl_j:>10.6f} {combined_kl:>10.6f} "
              f"{interaction:>+13.6f}  {classify(interaction)}  {format_eta(eta)}")

    # ── Build final matrix ─────────────────────────────────────────────────────
    print("\n  Building interaction matrix from all pair results...")
    matrix = build_interaction_matrix(layer_names, isolated_kl, pairwise_kl)
    np.save(MATRIX_FILE, matrix)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_min = (time.time() - scan_start) / 60

    # Gather all interaction values for ranking
    all_interactions = []
    for key, combined_kl in pairwise_kl.items():
        i, j        = map(int, key.split(','))
        name_i      = layer_names[i]
        name_j      = layer_names[j]
        interaction = combined_kl - isolated_kl.get(name_i, 0) - isolated_kl.get(name_j, 0)
        all_interactions.append((interaction, i, j, name_i, name_j))

    all_interactions.sort(reverse=True)

    n_super = sum(1 for v, *_ in all_interactions if v >  1e-4)
    n_sub   = sum(1 for v, *_ in all_interactions if v < -1e-4)
    n_indep = total_pairs - n_super - n_sub

    print(f"\n{'='*70}")
    print(f"  SCAN COMPLETE — {total_pairs} pairs | {total_min:.1f} minutes")
    print(f"{'='*70}")
    print(f"\n  Super-additive pairs (toxic)    : {n_super}")
    print(f"  Sub-additive pairs (synergistic): {n_sub}")
    print(f"  Independent pairs               : {n_indep}")

    print("\n  ── Top 5 Most TOXIC Pairs (errors compound — never quantize together) ──")
    for val, i, j, ni, nj in all_interactions[:5]:
        print(f"     {val:+.6f}  {shorten(ni)}  x  {shorten(nj)}")

    print("\n  ── Top 5 Most SYNERGISTIC Pairs (errors cancel — safe to quantize together) ──")
    for val, i, j, ni, nj in all_interactions[-5:]:
        print(f"     {val:+.6f}  {shorten(ni)}  x  {shorten(nj)}")

    # Write human-readable preview
    with open(PREVIEW_FILE, 'w', encoding='utf-8') as f:
        f.write("Pairwise Interaction Matrix — Summary\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Pairs: {total_pairs} | Super-additive: {n_super} | "
                f"Sub-additive: {n_sub} | Independent: {n_indep}\n\n")
        f.write("Top 5 Toxic Pairs (avoid quantizing together):\n")
        for val, i, j, ni, nj in all_interactions[:5]:
            f.write(f"  {val:+.6f}  L{i}:{ni}  x  L{j}:{nj}\n")
        f.write("\nTop 5 Synergistic Pairs (safe to quantize together):\n")
        for val, i, j, ni, nj in all_interactions[-5:]:
            f.write(f"  {val:+.6f}  L{i}:{ni}  x  L{j}:{nj}\n")

    print(f"\n  Interaction matrix  → {MATRIX_FILE}")
    print(f"  Checkpoint scores   → {PAIRWISE_FILE}")
    print(f"  Human preview       → {PREVIEW_FILE}")
    print(f"\n  Run step8_visualize_matrix.py next to plot the heatmap.\n")


if __name__ == "__main__":
    main()
