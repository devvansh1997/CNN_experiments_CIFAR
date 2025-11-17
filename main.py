import torch
import torch.nn as nn
import wandb
import yaml
import os

from utils.dataset import get_dataLoaders
from utils.train import train_one_epoch, evaluate
from models.cnn_large import CNN_Large
from models.cnn_small import CNN_Small

# Load Config
with open("config/default.yaml") as f:
    cfg = yaml.safe_load(f)
    print("----- Config Loaded -----")

# W & B
if cfg['log_wandb']:
    wandb.init(
        project=cfg['experiment_name'],
        name=cfg['run_name'],
        config=cfg
    )
    print("----- W&B Initialized -----")

# build data loaders
train_loader, test_loader = get_dataLoaders(
    batch_size=cfg['batch_size']
)
print("----- Dataset Loaded -----")

# intialize models

device = "mps" if torch.backends.mps.is_available() else "cpu"

if cfg["model"]["type"] == "cnn_small":
    model = CNN_Small().to(device)
elif cfg["model"]["type"] == "cnn_large":
    model = CNN_Large().to(device)
else:
    raise ValueError("Unknown model type")

# log param count
if cfg["log_wandb"]:
    wandb.watch(model, log="all", log_freq=100)

# initialize training params
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg["learning_rate"],
    momentum=cfg["momentum"],
    weight_decay=cfg["weight_decay"]
)

scheduler = None
if cfg["lr_scheduler"]["use"]:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["lr_scheduler"]["step_size"],
        gamma=cfg["lr_scheduler"]["gamma"]
    )

# start training
print("----- Begin Training -----")
best_acc = 0.0
os.makedirs(cfg['model']['save_path'], exist_ok=True)

for epoch in range(cfg["epochs"]):

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device
    )

    if scheduler:
        scheduler.step()

    # log to wandb
    wandb.log({
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "epoch": epoch + 1
    })

    print(f"Epoch {epoch+1}/{cfg['epochs']} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

    # Save best model
    
    if cfg["save_best_model"] and test_acc > best_acc:
        best_acc = test_acc
        save_path = f"{cfg['model']['save_path']}/{cfg['model']['type']}_best.pth"
        torch.save(model.state_dict(), save_path)

wandb.finish()
print("----- Training Complete -----")