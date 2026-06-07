
import torch

x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, device='cpu')
print(x)
print(x.shape)
print(x.dtype)
print(x.device)

x = torch.tensor([-2.0, 4.0, -6.0])

print(torch.abs(x))     # [2, 4, 6]
print(torch.mean(x))    # average value
print(x.shape)          # size of the tensor
print(x.numpy())        # converts to NumPy array

x = torch.tensor([5.5])
print(x.item())   # 5.5 as a Python float

x = torch.tensor([5.5, 7.2])
print(x[0].type)
print(x[0].item())
print(x[0])       # tensor(5.5000)
print(x[0].item())# 5.5

torch.nn.Linear(10, 5)

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

#for name, module in model.named_modules():
    #print(name, type(module))

#for name, param in model.named_parameters():
 #   print(name, param.shape)
    
#sd = model.state_dict()
#print(sd.keys())


logits = model(x)
preds = torch.argmax(logits, dim=1)
correct = (preds == y_true).sum()
accuracy = correct.item() / y_true.size(0)
print(accuracy)