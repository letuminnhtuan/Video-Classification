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
