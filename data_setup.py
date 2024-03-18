import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CustomDataset(Dataset):
    def __init__(self, root_path, file_csv):
        super(CustomDataset, self).__init__()
        self.root_path = root_path
        self.file_csv = file_csv
        # Get path of video & label
        df = pd.read_csv(self.file_csv)
        self.paths = df['paths'].tolist()
        self.labels = df['labels'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Get video_path
        tensor_path = self.paths[index]
        tensor = torch.load(tensor_path, map_location=device)
        # Get label
        output_label = self.labels[index]
        return tensor, torch.tensor(output_label)

def create_dataloader(root_path, train_path, val_path, batch_size):
    train_dataset = CustomDataset(root_path, train_path)
    val_dataset = CustomDataset(root_path, val_path)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    class_weights = torch.load('class_weights.pt', map_location=device)
    return train_loader, val_loader, class_weights