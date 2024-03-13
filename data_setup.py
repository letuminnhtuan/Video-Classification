import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

class CustomDataset(Dataset):
    def __init__(self, root_path, image_size, num_frames):
        super(CustomDataset, self).__init__()
        self.root_path = root_path
        self.image_size = image_size
        self.num_frames = num_frames
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(self.image_size, antialias=True),
            v2.Normalize([0.5], [0.5]),
        ])
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

    def get_frames(self, video_path):
        i = 0
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(np.asarray(frame))
            i += 1
            if i == self.num_frames:
                break
        frame = frames[-1]
        while i != self.num_frames:
            frames.append(np.zeros_like(frame))
            i += 1
        return frames

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # Get video_path
        video_path = self.paths[index]
        frames = self.get_frames(video_path)
        tensors = []
        for frame in frames:
            tensor = self.transform(frame)
            tensor = torch.unsqueeze(tensor, dim=0)
            tensors.append(tensor)
        input_video = torch.concatenate(tensors, dim=0)
        input_video = input_video.transpose(0, 1)
        # Get label
        output_label = self.labels[index]
        return input_video, torch.tensor(output_label)

def create_dataloader(root_path, image_size, num_frames, batch_size):
    dataset = CustomDataset(root_path, image_size, num_frames)
    dataloader = DataLoader(dataset, batch_size)
    return dataloader, dataset.classes_name