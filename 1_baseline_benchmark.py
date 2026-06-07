import time
import psutil
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def get_memory_usage():
    """Returns the current memory usage of this process in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("--- Step 1: Loading Baseline PyTorch Model ---")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Track memory before loading the model
mem_before = get_memory_usage()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Track memory after loading the model
mem_after = get_memory_usage()
model_memory = mem_after - mem_before
print(f"Model loaded into RAM. Memory consumed: {model_memory:.2f} MB\n")

# A sample evaluation dataset to simulate user requests
sample_texts = [
    "This hardware optimization approach is absolutely brilliant and game-changing.",
    "The execution was incredibly slow, inefficient, and a complete waste of time.",
    "It works okay, but the latency is highly unpredictable for production use.",
    "The software pipeline runs smoothly without any noticeable lag or overhead.",
    "I am highly disappointed with the computational constraints of this processor."
] * 20  # Expanded to 100 sentences to get a reliable average latency

print(f"--- Step 2: Benchmarking Inference on 100 Sentences (CPU) ---")

start_time = time.time()

# Tell PyTorch to disable gradient calculations (saves huge amounts of memory during inference)
with torch.no_grad():
    for text in sample_texts:
        # Tokenization: Convert text to raw tensor arrays
        inputs = tokenizer(text, return_tensors="pt")
        # Inference: Run the math forward through the network
        outputs = model(**inputs)

end_time = time.time()

total_time = end_time - start_time
average_latency = (total_time / len(sample_texts)) * 1000 # convert to milliseconds

print(f"Total time taken: {total_time:.4f} seconds")
print(f"Average latency per sentence: {average_latency:.2f} ms")
print("---------------------------------------------------------")