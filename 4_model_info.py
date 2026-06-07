from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Loop through named modules
for name, module in model.named_modules():
    # We only care about the actual mathematical layers doing the matrix math
    if "Linear" in str(type(module)):
        print(f"Layer Found: {name} | Type: {type(module)}")