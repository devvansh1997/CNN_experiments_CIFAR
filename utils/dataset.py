import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# build data loaders
def get_dataLoaders(batch_size = 128, data_dir = "./data"):

    # pre-computed std dev and means (pulled directly from a web search)
    CIFAR_10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_10_STD = (0.2470, 0.2435, 0.2616)

    # transforms
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_10_MEAN, CIFAR_10_STD)
    ])

    # dataset objects
    train_ds = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transforms
    )

    test_ds = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transforms
    )

    # data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, test_loader