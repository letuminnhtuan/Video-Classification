import torch
from model_builder import ViT
from data_setup import create_dataloader
from metrics import accuracy_fn, EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""" Create dataloader """
root_path = 'data/'
image_size = (224, 224)
num_frames = 48
batch_size = 32
dataloader, classes = create_dataloader(root_path, image_size, num_frames, batch_size)

""" Create Model """
input_chanel = 3
output_chanel = 32
n_head = 1
n_expansion = 4
n_layer = 1
num_classes = len(classes)
model = ViT(input_chanel, output_chanel, n_head, n_expansion, n_layer, num_classes).to(device)

""" Create Metrics """
epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=5, delta=0.001, verbose=True)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

""" Training model """
for epoch in range(epochs):
    # ---------------------------------------- Training ----------------------------------------
    train_loss = 0
    train_acc = 0
    for batch, (X, y) in enumerate(dataloader):
        model.train()
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    # ---------------------------------------- Validation ----------------------------------------
    val_loss = 0
    val_acc = 0
    # model.eval()
    # with torch.no_grad():
    #     for batch, (X, y) in enumerate(dataloader):
    #         X, y = X.to(device), y.to(device)
    #         # Forward pass
    #         y_pred = model(X)
    #         # Calculate validation loss and accuracy
    #         val_loss += criterion(y_pred, y).item()
    #         val_acc += accuracy_fn(torch.softmax(y_pred, dim=1).argmax(dim=1), y)
    # val_loss /= len(dataloader)
    # val_acc /= len(dataloader)
    print(f"Epoch: {epoch + 1}/{epochs} - Training Loss: {train_loss:.2f} - Training Accuracy: {train_acc:.2f} - Validation Loss: {val_loss:.2f} - Validation Accuracy: {val_acc:.2f}")
    # ---------------------------------------- Learning rate Schedule ----------------------------------------
    lr_scheduler.step(val_loss)
    # ---------------------------------------- Check EarlyStopping ----------------------------------------
    if early_stopping(val_loss):
        break
