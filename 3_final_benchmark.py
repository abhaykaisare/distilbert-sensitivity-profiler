import time
import os
import psutil
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# 1. Setup Data
sample_texts = [
    "This hardware optimization approach is absolutely brilliant and game-changing.",
    "The execution was incredibly slow, inefficient, and a complete waste of time.",
    "It works okay, but the latency is highly unpredictable for production use.",
    "The software pipeline runs smoothly without any noticeable lag or overhead.",
    "I am highly disappointed with the computational constraints of this processor."
] * 20  # 100 total test inputs

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("=== RUNNING BASELINE PYTORCH BENCHMARK ===")
mem_start_pt = get_memory_usage()
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
mem_loaded_pt = get_memory_usage()

pt_predictions = []
start_time = time.time()
with torch.no_grad():
    for text in sample_texts:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = pt_model(**inputs)
        # Get the highest probability class (0 = Negative, 1 = Positive)
        pred = torch.argmax(outputs.logits, dim=1).item()
        pt_predictions.append(pred)
pt_time = time.time() - start_time
pt_latency = (pt_time / len(sample_texts)) * 1000
pt_ram = mem_loaded_pt - mem_start_pt

# Clean up memory explicitly to isolate the ONNX test
del pt_model
if torch.cuda.is_available(): torch.cuda.empty_cache()

print("\n=== RUNNING OPTIMIZED INT8 ONNX BENCHMARK ===")
quantized_model_path = os.path.join("onnx_model", "model_quantized.onnx")

# Setup ONNX Runtime to use your Intel CPU efficiently
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

mem_start_onnx = get_memory_usage()
# Initialize the runtime session with our INT8 model
ort_session = ort.InferenceSession(quantized_model_path, session_options, providers=['CPUExecutionProvider'])
mem_loaded_onnx = get_memory_usage()

onnx_predictions = []
start_time = time.time()

for text in sample_texts:
    # ONNX expects raw numpy arrays, not PyTorch tensors
    inputs = tokenizer(text, return_tensors="np")
    # Structure inputs exactly as the ONNX graph expects them
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    outputs = ort_session.run(None, onnx_inputs)
    # Get predictions out of the ONNX array math
    pred = np.argmax(outputs[0], axis=1)[0]
    onnx_predictions.append(pred)

onnx_time = time.time() - start_time
onnx_latency = (onnx_time / len(sample_texts)) * 1000
onnx_ram = mem_loaded_onnx - mem_start_onnx

# 3. Analyze Alignment (Accuracy Retention)
matches = sum(1 for p, o in zip(pt_predictions, onnx_predictions) if p == o)
alignment_score = (matches / len(sample_texts)) * 100

print("\n================ FINAL REPORT ================")
print(f"Baseline PyTorch Latency : {pt_latency:.2f} ms per sentence")
print(f"Optimized INT8 ONNX Latency: {onnx_latency:.2f} ms per sentence")
print(f"--> LATENCY SPEEDUP FACTOR : {pt_latency / onnx_latency:.2f}x Faster\n")

print(f"Baseline PyTorch RAM Footprint : {pt_ram:.2f} MB")
print(f"Optimized INT8 ONNX RAM Footprint: {onnx_ram:.2f} MB")
print(f"--> RAM REDUCTION FACTOR       : {pt_ram - onnx_ram:.2f} MB Saved\n")

print(f"Model Output Alignment (Accuracy Check): {alignment_score:.2f}%")
print("==============================================")