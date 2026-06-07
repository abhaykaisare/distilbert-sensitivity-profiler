# DistilBERT Layer Quantization Sensitivity Profiler

A diagnostic tool that measures how sensitive each individual layer of DistilBERT is to INT8 quantization, then compiles a mixed-precision model using those measurements.

Built entirely on CPU. No GPU required.

---

![Sensitivity Map](sensitivity_map.png)

*Each bar shows how much quantizing that single layer disturbs the model's output distribution (KL divergence). Red = sensitive, keep at FP32. Green = resilient, safe to compress to INT8.*

---

## Motivation

Most quantization tools apply INT8 uniformly across the entire model. But not all layers deserve the same treatment — some are highly sensitive to precision loss, others are nearly unaffected. Blind compression trades accuracy for speed without knowing which layers are actually responsible for the tradeoff.

This project builds the measurement layer: a profiler that isolates each layer, quantizes it individually, and records the resulting distribution shift. The output is an evidence-based mixed-precision model, not a guess.

A secondary goal was accessibility. This runs entirely on a CPU. If you have a laptop, you can run, modify, and extend this.

---

## Key Results

| Model | Accuracy | Latency | Size | INT8 Layers |
|---|---|---|---|---|
| FP32 Baseline | 91.25% | 22.77 ms | 255.5 MB | 0 / 38 |
| Full INT8 (blind) | 90.50% | 9.20 ms | 132.3 MB | 38 / 38 |
| Mixed Precision (ours) | **91.25%** | 21.26 ms | 218.3 MB | 17 / 38 |

**Mixed precision achieved 0% accuracy drop** vs 0.75% for blind INT8, by protecting the 21 layers the profiler identified as sensitive.

The speedup was modest (1.07x). That result is honest and worth understanding — see [What the results reveal](#what-the-results-reveal).

---

## How It Works

The pipeline runs in 6 steps:

```
step1  →  Map every Linear layer in the model
step2  →  Prove single-layer quantization + KL metric works
step3  →  Scan all 38 layers, measure KL divergence per layer  (~10 min)
step4  →  Plot the sensitivity map
step5  →  Compile the mixed-precision model
step6  →  Run inference demo
```

**Why KL Divergence, not accuracy?**
Hard accuracy was the first metric tried. It failed: on a well-trained 2-class model, quantization shifts logits slightly but rarely flips a prediction, so accuracy stayed frozen at 91.25% regardless of which layer was touched. KL divergence between the original and quantized output distributions is continuous — it detects distribution shift even when no prediction changes. That switch made the profiler work.

---

## What the Results Reveal

The top 5 most sensitive layers are all FFN layers (`lin1`, `lin2`). The top 5 most resilient are mostly final-block attention layers and the classifier head.

Two effects drive this:

**Error compounding.** Quantization error introduced at layer 1 propagates through 5 subsequent layers before reaching the output. Layer 5's error reaches the output in one step. Early layers are more sensitive because their errors amplify.

**Layer size.** FFN layers are 3072×768 = 2.3M parameters. Attention layers are 768×768 = 590K. More parameters means more total quantization error summed across the matrix.

The mixed-precision speedup was limited to 1.07x because the layers the profiler correctly identified as sensitive (FFN blocks 0–4) are also the most compute-expensive layers. Protecting accuracy and gaining speed require protecting the same layers — this is the core tension that systems like [HAWQ](https://arxiv.org/abs/1905.03696) and [SmoothQuant](https://arxiv.org/abs/2211.10438) were designed to navigate beyond a binary protect/quantize decision.

---

## Installation

```bash
git clone https://github.com/<your-username>/distilbert-sensitivity-profiler
cd distilbert-sensitivity-profiler
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, no GPU needed.

---

## Usage

Run the steps in order. Each step produces output that the next one reads.

```bash
# Step 1 — map all layers, generates layer_names.txt
python step1_explore_model.py

# Step 2 — test the core mechanism on a single layer
python step2_single_layer_quant.py

# Step 3 — full scan (~10 min on CPU, saves after each layer)
python step3_sensitivity_scan.py

# Step 4 — generate sensitivity_map.png
python step4_plot_map.py

# Step 5 — compile and benchmark the mixed-precision model
python step5_mixed_precision_compiler.py

# Step 6 — inference demo
python step6_final_demo.py
python step6_final_demo.py "Your custom sentence here."
```

**Step 3 is resumable.** Results are saved to `sensitivity_results.json` after every layer. If the script is interrupted, re-running it skips completed layers automatically.

---

## Project Structure

```
├── step1_explore_model.py        # Layer map
├── step2_single_layer_quant.py   # Single-layer probe with KL metric
├── step3_sensitivity_scan.py     # Full automated scan
├── step4_plot_map.py             # Sensitivity chart
├── step5_mixed_precision_compiler.py  # Mixed-precision model builder + benchmark
├── step6_final_demo.py           # Inference demo
│
├── sensitivity_results.json      # Scan output (38 layers, KL + MSE scores)
├── sensitivity_map.png           # The chart
├── requirements.txt
│
├── 1_baseline_benchmark.py       # Original ONNX baseline (FP32 reference)
├── 2_optimize_and_quantize.py    # ONNX full-model INT8 quantization
└── 3_final_benchmark.py          # ONNX benchmark (8.28x speedup reference)
```

The files numbered 1–3 are the original ONNX quantization baseline that preceded this project. They provide the reference point: standard ONNX dynamic INT8 achieves 8.28x speedup on this model. The sensitivity profiler then explains *why* that approach loses accuracy and *which* layers are responsible.

---

## A Note on Existing Tools

ONNX Runtime already does quantization, and does it more efficiently than this. This project makes no claim to replace it. The goal was to build the measurement layer from scratch — to understand what uniform quantization is actually doing to each layer, and whether a smarter allocation of precision budget changes the outcome. That required writing the profiler by hand rather than calling a high-level API.

---

## What's Next

The natural extension is replacing the binary protect/quantize decision with learnable per-layer quantization scales — letting a calibration pass determine the optimal precision for each layer rather than using a fixed KL threshold. This is what HAWQ and similar methods implement.

The profiler generalises to any HuggingFace sequence classification model. Swap `MODEL_NAME` in any step to run the full pipeline on a different architecture.

---

## References

- [HAWQ: Hessian AWare Quantization](https://arxiv.org/abs/1905.03696)
- [SmoothQuant](https://arxiv.org/abs/2211.10438)
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)
- [The Illustrated BERT — Jay Alammar](https://jalammar.github.io/illustrated-bert/)
