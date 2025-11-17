import torch

def train_one_epoch(model, dataloader, loss_fn, optimzer, device):

    # put the model in training mode
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # training loop
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # backward pass
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        # track loss
        running_loss += loss.item()

        # track accuracy
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):

    # put model in eval mode
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # get predictions
        outputs = model(images)

        # compute loss
        loss = loss_fn(outputs, labels)

        # track loss
        running_loss += loss.item()

        # track accuracy
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy