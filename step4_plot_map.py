"""
step4_plot_map.py
-----------------
Purpose: Read sensitivity_results.json from step3 and generate the
layer sensitivity map chart. This is your primary shareable output.

Output:
  sensitivity_map.png  — the chart (post on LinkedIn, add to README)
  sensitivity_summary.txt — fixed version (UTF-8, no encoding crash)

Run: python step4_plot_map.py
Requires: sensitivity_results.json from step3
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

RESULTS_FILE  = "sensitivity_results.json"
OUTPUT_CHART  = "sensitivity_map.png"
SUMMARY_FILE  = "sensitivity_summary.txt"
KL_THRESHOLD  = 1e-4

# ── Helpers ────────────────────────────────────────────────────────────────────

def abbreviate(name: str) -> str:
    """
    Shortens full layer names to fit on the chart x-axis.
    'distilbert.transformer.layer.2.ffn.lin1' → 'L2-ffn\n1'
    'pre_classifier' → 'pre_cls'
    """
    if name == 'pre_classifier': return 'pre_cls'
    if name == 'classifier':     return 'cls'
    parts = name.split('.')
    lnum  = parts[3]
    ltype = parts[4]
    sub   = parts[5]
    if ltype == 'attention':
        short = {'q_lin': 'q', 'k_lin': 'k', 'v_lin': 'v', 'out_lin': 'out'}
        return f"L{lnum}-attn\n{short.get(sub, sub)}"
    return f"L{lnum}-ffn\n{sub.replace('lin','')}"


def plot_sensitivity_map(results: dict):
    """
    Generates a log-scale bar chart of KL divergence per layer.

    Why log scale:
      The KL range spans 6 orders of magnitude — from ~5e-8 (classifier)
      to ~0.027 (layer.1.ffn.lin2). On a linear scale, the small bars
      would be completely invisible. Log scale shows the full range.

    Color coding:
      Red  = KL > threshold → keep this layer at FP32
      Green = KL ≤ threshold → safe to quantize to INT8
    """
    sorted_layers = sorted(results.items(), key=lambda x: x[1]['layer_index'])

    names     = [abbreviate(n) for n, _ in sorted_layers]
    kl_scores = [r['kl_divergence'] for _, r in sorted_layers]
    colors    = ['#C0392B' if kl > KL_THRESHOLD else '#27AE60' for kl in kl_scores]

    n_sensitive = sum(1 for kl in kl_scores if kl > KL_THRESHOLD)
    n_resilient = len(kl_scores) - n_sensitive

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8F9FA')

    # ── Bars ──────────────────────────────────────────────────────────────────
    ax.bar(range(len(names)), kl_scores,
           color=colors, edgecolor='white', linewidth=0.6,
           alpha=0.88, width=0.72, zorder=3)

    # ── Log scale ─────────────────────────────────────────────────────────────
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10))

    # ── Threshold line ────────────────────────────────────────────────────────
    ax.axhline(y=KL_THRESHOLD, color='#E67E22', linestyle='--',
               linewidth=1.8, alpha=0.9, zorder=4, label=f'Threshold (1e-4)')

    # ── Subtle grid ───────────────────────────────────────────────────────────
    ax.yaxis.grid(True, alpha=0.25, linestyle='-', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── Transformer block separators and labels ───────────────────────────────
    # DistilBERT has 6 transformer blocks, each with 6 Linear layers.
    # The last 2 entries are the classifier head.
    for block in range(1, 6):
        ax.axvline(x=block * 6 - 0.5, color='#BDC3C7',
                   linestyle=':', linewidth=1, alpha=0.7, zorder=2)

    ylim_top = max(kl_scores) * 20
    block_labels    = ['Block 0', 'Block 1', 'Block 2', 'Block 3', 'Block 4', 'Block 5', 'Head']
    block_positions = [2.5, 8.5, 14.5, 20.5, 26.5, 32.5, 36.5]
    for bpos, bname in zip(block_positions, block_labels):
        ax.text(bpos, ylim_top, bname, ha='center', va='top',
                fontsize=8, color='#7F8C8D', fontstyle='italic')

    # ── Annotate top 3 most sensitive layers ──────────────────────────────────
    top3    = sorted(enumerate(kl_scores), key=lambda x: x[1], reverse=True)[:3]
    offsets = [(4, 3.5), (-6, 3.0), (-7, 2.5)]
    for rank, ((idx, kl), (dx, dy)) in enumerate(zip(top3, offsets)):
        label = names[idx].replace('\n', '-')
        ax.annotate(
            f"#{rank+1}  {label}\nKL={kl:.4f}",
            xy=(idx, kl),
            xytext=(idx + dx, kl * dy),
            fontsize=8, color='#922B21', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#922B21', lw=1.3),
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#FDEDEC', edgecolor='#C0392B', alpha=0.85)
        )

    # ── Axes labels and title ─────────────────────────────────────────────────
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5, ha='center')
    ax.set_xlabel('Layer  (grouped by transformer block)',
                  fontsize=11, labelpad=12, color='#2C3E50')
    ax.set_ylabel('KL Divergence  (log scale)',
                  fontsize=11, labelpad=10, color='#2C3E50')
    ax.set_title(
        'DistilBERT — Layer Quantization Sensitivity Map\n'
        'KL divergence between FP32 and INT8 output distributions  |  SST-2 validation  |  n=400',
        fontsize=13, fontweight='bold', pad=18, color='#1A252F'
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    p_s = mpatches.Patch(color='#C0392B', alpha=0.88,
                         label=f'Sensitive (KL > 1e-4) — keep FP32 : {n_sensitive} layers')
    p_r = mpatches.Patch(color='#27AE60', alpha=0.88,
                         label=f'Resilient (KL <= 1e-4) — safe to quantize : {n_resilient} layers')
    ax.legend(handles=[p_s, p_r, ax.get_lines()[0]],
              fontsize=9, loc='upper right', framealpha=0.92,
              edgecolor='#BDC3C7', fancybox=True)

    # ── Clean spines ──────────────────────────────────────────────────────────
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')

    plt.tight_layout()
    plt.savefig(OUTPUT_CHART, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved to: {OUTPUT_CHART}")


def write_summary(results: dict):
    """
    Writes a human-readable ranked summary.
    Uses UTF-8 encoding explicitly to avoid Windows cp1252 crash.
    Uses plain '-' separators instead of box-drawing characters.
    """
    ranked = sorted(results.items(),
                    key=lambda x: x[1]['kl_divergence'], reverse=True)

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write("DistilBERT SST-2 - Layer Sensitivity Scan Results\n")
        f.write(f"Baseline accuracy : {ranked[0][1]['acc_fp32'] * 100:.2f}%\n")
        f.write(f"KL threshold      : {KL_THRESHOLD}\n\n")
        f.write(f"{'Rank':<6}{'Layer Name':<60}{'KL Score':<14}{'MSE':<14}Verdict\n")
        f.write("-" * 100 + "\n")
        for rank, (name, r) in enumerate(ranked, 1):
            verdict = "SENSITIVE" if r['kl_divergence'] > KL_THRESHOLD else "RESILIENT"
            f.write(f"{rank:<6}{name:<60}{r['kl_divergence']:<14.6f}{r['mse']:<14.6f}{verdict}\n")

    print(f"  Summary saved to: {SUMMARY_FILE}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  STEP 4: Plotting Sensitivity Map")
    print("=" * 55)

    if not os.path.exists(RESULTS_FILE):
        print(f"\n  ERROR: {RESULTS_FILE} not found.")
        print(f"  Run step3_sensitivity_scan.py first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)

    print(f"\n  Loaded {len(results)} layer results.")

    print("  Generating chart...")
    plot_sensitivity_map(results)

    print("  Writing summary...")
    write_summary(results)

    print(f"\n  Done. Open {OUTPUT_CHART} to see your sensitivity map.")
    print(f"  This chart is your LinkedIn post visual.\n")


if __name__ == "__main__":
    main()
