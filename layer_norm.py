import torch
import torch.nn as nn
import random

torch.manual_seed(43)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
std = out.std(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("----> mean -->  ", mean)
print("----> std -->  ", std)
print("----> var -->  ", var)

# layer norm using torch
layer_norm = nn.LayerNorm(6)
out_torch = layer_norm(out)
print("----torch layer norm----")
print(out)


# layer norm using formula
out_manual = (out - mean) / std
print("----manual layer norm----")
print(out)
print("----> mean --> ", out_manual.mean(dim=-1, keepdim=True))
print("----> std --> ", out_manual.std(dim=-1, keepdim=True))
print("----> var --> ", out_manual.var(dim=-1, keepdim=True))




