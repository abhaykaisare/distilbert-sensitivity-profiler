"""
step8_visualize_matrix.py
--------------------------
Purpose: Load the pairwise interaction data from step7 and plot the
38x38 interaction matrix as a heatmap.

Reading the chart:
  Red cell  (i,j) = super-additive: quantizing layers i and j together
                    causes MORE damage than the sum of their individual scores
  Blue cell (i,j) = sub-additive:  quantizing both causes LESS damage
                    than predicted — errors partially cancel
  White cell      = independent:   step5's assumption holds for this pair

  Stars (★) mark the top 5 most toxic pairs.
  Triangles (△) mark the top 5 most synergistic pairs.

Output: interaction_heatmap.png

Requires: pairwise_kl_scores.json (step7), sensitivity_results.json (step3),
          layer_names.txt (step1)
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

PAIRWISE_FILE  = "pairwise_kl_scores.json"
RESULTS_FILE   = "sensitivity_results.json"
LAYER_FILE     = "docs/layer_names.txt"
OUTPUT_FILE    = "interaction_heatmap.png"
KL_THRESHOLD   = 1e-4   # interaction threshold for classifying pairs


# ── Helpers ────────────────────────────────────────────────────────────────────

def shorten(name):
    if name == 'pre_classifier': return 'pre_cls'
    if name == 'classifier':     return 'cls'
    parts = name.split('.')
    lnum  = parts[3]; ltype = parts[4]; sub = parts[5]
    if ltype == 'attention':
        m = {'q_lin': 'q', 'k_lin': 'k', 'v_lin': 'v', 'out_lin': 'out'}
        return f"L{lnum}.a.{m.get(sub, sub)}"
    return f"L{lnum}.f.{sub.replace('lin', '')}"


def build_matrix(layer_names, isolated_kl, pairwise_kl):
    """Reconstructs the full interaction matrix from stored pair scores."""
    n      = len(layer_names)
    matrix = np.zeros((n, n))
    for key, combined_kl in pairwise_kl.items():
        i, j   = map(int, key.split(','))
        ni, nj = layer_names[i], layer_names[j]
        val    = combined_kl - isolated_kl.get(ni, 0) - isolated_kl.get(nj, 0)
        matrix[i][j] = val
        matrix[j][i] = val
    return matrix


def plot_heatmap(matrix, layer_names, isolated_kl, pairwise_kl):
    n      = len(layer_names)
    labels = [shorten(name) for name in layer_names]

    # Collect all interactions for ranking
    all_vals = []
    for key, ckl in pairwise_kl.items():
        i, j   = map(int, key.split(','))
        ni, nj = layer_names[i], layer_names[j]
        val    = ckl - isolated_kl.get(ni, 0) - isolated_kl.get(nj, 0)
        all_vals.append((val, i, j, shorten(ni), shorten(nj)))
    all_vals.sort(reverse=True)

    n_super = sum(1 for v, *_ in all_vals if v >  KL_THRESHOLD)
    n_sub   = sum(1 for v, *_ in all_vals if v < -KL_THRESHOLD)
    n_ind   = len(all_vals) - n_super - n_sub

    # ── Figure layout: heatmap (left) + summary panel (right) ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(22, 9),
                              gridspec_kw={'width_ratios': [2.2, 1]})
    fig.patch.set_facecolor('#FFFFFF')

    # ── Heatmap ───────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#F8F9FA')

    # Symmetric color scale centered at 0
    # Clamp to 60% of max range so mid-range values are still visible
    abs_max = max(abs(matrix.max()), abs(matrix.min()))
    vmax    =  abs_max * 0.6
    vmin    = -abs_max * 0.6

    im = ax.imshow(matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    # Transformer block separators (every 6 layers = one block)
    for b in range(1, 6):
        ax.axhline(b * 6 - 0.5, color='#2C3E50', linewidth=1.2, alpha=0.6)
        ax.axvline(b * 6 - 0.5, color='#2C3E50', linewidth=1.2, alpha=0.6)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticklabels(labels, fontsize=6.5)

    # Block header labels
    block_centers = [2.5, 8.5, 14.5, 20.5, 26.5, 32.5, 36.5]
    block_labels  = ['Blk0','Blk1','Blk2','Blk3','Blk4','Blk5','Head']
    for bc, bl in zip(block_centers, block_labels):
        ax.text(bc, -2.3, bl, ha='center', va='bottom', fontsize=7.5,
                color='#34495E', fontweight='bold')
        ax.text(-2.9, bc, bl, ha='right',  va='center', fontsize=7.5,
                color='#34495E', fontweight='bold')

    # Mark top 5 toxic pairs with black stars
    for val, i, j, *_ in all_vals[:5]:
        ax.plot(j, i, 'k*', markersize=8, alpha=0.95, zorder=5)
        ax.plot(i, j, 'k*', markersize=8, alpha=0.95, zorder=5)

    # Mark top 5 synergistic pairs with white triangles
    for val, i, j, *_ in all_vals[-5:]:
        ax.plot(j, i, 'w^', markersize=6, alpha=0.95, zorder=5)
        ax.plot(i, j, 'w^', markersize=6, alpha=0.95, zorder=5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Interaction  =  KL(both)  −  KL(i)  −  KL(j)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        'DistilBERT — Layer Pair Interaction Matrix\n'
        'Red = super-additive (toxic)   Blue = sub-additive (synergistic)   White = independent\n'
        '★ = top 5 toxic pairs     △ = top 5 synergistic pairs',
        fontsize=11, fontweight='bold', pad=14, color='#1A252F'
    )

    # ── Summary panel ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#F8F9FA')
    ax2.axis('off')

    # Pair counts
    stats = (
        f"Pairs Measured: {len(all_vals)}\n\n"
        f"  Super-additive (toxic)\n"
        f"  {n_super} pairs  ({n_super/len(all_vals)*100:.1f}%)\n\n"
        f"  Sub-additive (synergistic)\n"
        f"  {n_sub} pairs  ({n_sub/len(all_vals)*100:.1f}%)\n\n"
        f"  Independent\n"
        f"  {n_ind} pairs  ({n_ind/len(all_vals)*100:.1f}%)\n"
    )
    ax2.text(0.05, 0.97, stats, transform=ax2.transAxes,
             fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#ECF0F1',
                       edgecolor='#BDC3C7', alpha=0.9))

    # Top 5 toxic
    ax2.text(0.05, 0.60, 'Top 5 Toxic Pairs  ★',
             transform=ax2.transAxes, fontsize=9.5,
             fontweight='bold', color='#C0392B')
    for k, (val, i, j, si, sj) in enumerate(all_vals[:5]):
        y = 0.54 - k * 0.065
        ax2.text(0.05, y, f"  {si}  x  {sj}",
                 transform=ax2.transAxes, fontsize=7.8,
                 color='#922B21', fontfamily='monospace')
        ax2.text(0.90, y, f"+{val:.4f}",
                 transform=ax2.transAxes, fontsize=7.8,
                 color='#C0392B', ha='right', fontfamily='monospace')

    # Top 5 synergistic
    ax2.text(0.05, 0.26, 'Top 5 Synergistic Pairs  △',
             transform=ax2.transAxes, fontsize=9.5,
             fontweight='bold', color='#1A5276')
    for k, (val, i, j, si, sj) in enumerate(reversed(all_vals[-5:])):
        y = 0.20 - k * 0.065
        ax2.text(0.05, y, f"  {si}  x  {sj}",
                 transform=ax2.transAxes, fontsize=7.8,
                 color='#1A5276', fontfamily='monospace')
        ax2.text(0.90, y, f"{val:.4f}",
                 transform=ax2.transAxes, fontsize=7.8,
                 color='#2980B9', ha='right', fontfamily='monospace')

    # Key insight callout at bottom
    ax2.text(0.05, 0.02,
             "Note: L17 (layer.2.ffn.lin2)\n"
             "appears in 3 of 5 toxic pairs\n"
             "AND the most synergistic pair.\n"
             "Partner determines behavior.",
             transform=ax2.transAxes, fontsize=8.5, color='#444444',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF9E7',
                       edgecolor='#F0B429', alpha=0.9))

    plt.tight_layout(pad=1.5)
    plt.savefig(OUTPUT_FILE, dpi=170, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved to: {OUTPUT_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 8: Visualize Interaction Matrix")
    print("=" * 55)

    for fname in [PAIRWISE_FILE, RESULTS_FILE, LAYER_FILE]:
        if not os.path.exists(fname):
            print(f"\n  ERROR: {fname} not found.")
            return

    with open(PAIRWISE_FILE,  'r') as f: pairwise_kl  = json.load(f)
    with open(RESULTS_FILE,   'r') as f: step3        = json.load(f)
    with open(LAYER_FILE,     'r') as f:
        layer_names = [l.strip() for l in f if l.strip()]

    isolated_kl = {name: step3[name]['kl_divergence']
                   for name in layer_names if name in step3}

    print(f"\n  Loaded {len(pairwise_kl)} pair scores.")
    print(f"  Building interaction matrix...")
    matrix = build_matrix(layer_names, isolated_kl, pairwise_kl)

    print(f"  Plotting heatmap...")
    plot_heatmap(matrix, layer_names, isolated_kl, pairwise_kl)
    print(f"\n  Done. Open {OUTPUT_FILE} to see the matrix.\n")


if __name__ == "__main__":
    main()
