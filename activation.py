import torch
import torch.nn as nn

# Gelu activation function
gelu = nn.GELU()
input = torch.randn(2, 5)
print(input)
output = gelu(input)
print(output)