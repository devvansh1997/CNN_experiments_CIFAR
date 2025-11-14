import torch
import matplotlib.pyplot as plt
from utils.dataset import get_dataLoaders

# get data loaders back
train_loader, test_loader = get_dataLoaders(batch_size=16)

# get first iterable from data loader
images, labels = next(iter(train_loader))
print("Images:", images.shape)
print("Labels:", labels)

# show a grid
img_grid = images[:8]  # first 8 images

# unnormalize for visualization
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)
img_grid = img_grid * std + mean

fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i, ax in enumerate(axes.flat):
    img = img_grid[i].permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.axis("off")
plt.show()