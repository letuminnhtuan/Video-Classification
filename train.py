import torch
import matplotlib.pyplot as plt
from model_builder import ViT
from data_setup import create_dataloader
from metrics import EarlyStopping
from engine import train_fn, val_fn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
""" Create dataloader """
root_path = 'processed_data/'
train_ratio = 0.8
batch_size = 64
train_loader, val_loader, classes = create_dataloader(root_path, train_ratio, batch_size)

""" Create Model """
input_chanel = 3
output_chanel = 32
n_head = 1
n_expansion = 4
n_layer = 1
num_classes = len(classes)
model = ViT(input_chanel, output_chanel, n_head, n_expansion, n_layer, num_classes).to(device)

""" Create Metrics """
epochs = 60
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
early_stopping = EarlyStopping(patience=5, delta=0.01, verbose=True)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=True)

""" Training model """
result = {
    'train_acc': [],
    'train_loss': [],
    'val_acc': [],
    'val_loss': []
}
for epoch in range(epochs):
    # ---------------------------------------- Training ----------------------------------------
    train_loss, train_acc = train_fn(model, optimizer, criterion, train_loader, device)
    # ---------------------------------------- Validation ----------------------------------------
    val_loss, val_acc = val_fn(model, criterion, val_loader, device)
    print(f"Epoch: {epoch + 1}/{epochs} - Training Loss: {train_loss:.2f} - Training Accuracy: {train_acc:.2f} - Validation Loss: {val_loss:.2f} - Validation Accuracy: {val_acc:.2f}")
    result['train_acc'].append(train_acc)
    result['train_loss'].append(train_loss)
    result['val_acc'].append(val_acc)
    result['val_loss'].append(val_loss)
    # ---------------------------------------- Learning rate Schedule ----------------------------------------
    lr_scheduler.step(val_loss)
    # ---------------------------------------- Check EarlyStopping ----------------------------------------
    if early_stopping(val_loss):
        break

plt.figure(1, figsize=(10, 10))
plt.plot(result['train_acc'])
plt.plot(result['val_acc'])
plt.legend(['train', 'val'])
plt.show()

plt.figure(2, figsize=(10, 10))
plt.plot(result['train_loss'])
plt.plot(result['val_loss'])
plt.legend(['train', 'val'])
plt.show()