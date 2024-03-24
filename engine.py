import torch
from metrics import accuracy_fn

def train_fn(model, optimizer, criterion, train_loader, device):
    train_loss = 0
    train_acc = 0
    for batch, (X, y) in enumerate(train_loader):
        model.train()
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    return train_loss, train_acc

def val_fn(model, criterion, val_loader, device):
    val_loss = 0
    val_acc = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            # Forward pass
            y_pred = model(X)
            # Calculate validation loss and accuracy
            val_loss += criterion(y_pred, y).item()
            val_acc += accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc