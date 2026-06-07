import os
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
onnx_folder = "onnx_model"
quantized_model_path = os.path.join(onnx_folder, "model_quantized.onnx")

print("--- Step 1: Exporting PyTorch Model to Standard ONNX format ---")
# This loads the model from your local cache and converts it to a standard ONNX graph
model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the raw, unquantized ONNX model to a folder
model.save_pretrained(onnx_folder)
tokenizer.save_pretrained(onnx_folder)
print(f"Standard ONNX model saved to folder: {onnx_folder}\n")


print("--- Step 2: Applying Dynamic INT8 Quantization ---")
# The path to the raw ONNX model file we just created
raw_onnx_model_path = os.path.join(onnx_folder, "model.onnx")

# This optimizes the graph weights specifically for CPU execution
quantize_dynamic(
    model_input=raw_onnx_model_path,      # Input: 32-bit float model
    model_output=quantized_model_path,    # Output: 8-bit integer model
    weight_type=QuantType.QInt8           # Force weights to 8-bit Signed Integers
)

print(f"Quantization Complete! Optimized model saved at: {quantized_model_path}")

# Let's compare the file sizes on your hard drive
original_size = os.path.getsize(raw_onnx_model_path) / (1024 * 1024)
quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)

print("\n--- Hardware Footprint Analysis ---")
print(f"Original ONNX Model Size:  {original_size:.2f} MB")
print(f"Quantized INT8 Model Size: {quantized_size:.2f} MB")
print(f"Storage reduction:{((original_size - quantized_size) / original_size) * 100:.2f}%")