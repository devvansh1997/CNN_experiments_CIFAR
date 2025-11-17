import torch
from models.cnn_small import CNN_Small
from models.cnn_large import CNN_Large

# sample input
x = torch.randn(4, 3, 32, 32)

# Small Model Test
model_small = CNN_Small()
out_small = model_small(x)
print(f"Small Output Shape -> {out_small.shape}")

# Large Model Test
model_large = CNN_Large()
out_large = model_large(x)
print(f"Large Output Shape -> {out_small.shape}")