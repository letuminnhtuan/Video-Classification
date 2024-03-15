import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CustomDataset(Dataset):
    def __init__(self, root_path):
        super(CustomDataset, self).__init__()
        self.root_path = root_path
        # Get path of video & label
        self.paths = []
        self.labels = []
        self.classes_name = []
        for idx, dir in enumerate(os.listdir(self.root_path)):
            video_folder = os.path.join(self.root_path, dir)
            self.classes_name.append(dir)
            for path in os.listdir(video_folder):
                video_path = os.path.join(video_folder, path)
                self.paths.append(video_path)
                self.labels.append(idx)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Get video_path
        tensor_path = self.paths[index]
        tensor = torch.load(tensor_path, map_location=device)
        # Get label
        output_label = self.labels[index]
        return tensor, torch.tensor(output_label)

def create_dataloader(root_path, train_ratio, batch_size):
    dataset = CustomDataset(root_path)
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    return train_loader, val_loader, dataset.classes_name