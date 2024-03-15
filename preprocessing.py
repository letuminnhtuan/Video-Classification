import os
import torch
import cv2
import numpy as np
import pandas as pd
from torchvision.transforms import v2

def get_frames(video_path, num_frames):
    i = 0
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(np.asarray(frame))
        i += 1
        if i == num_frames:
            break
    frame = frames[-1]
    while i != num_frames:
        frames.append(np.zeros_like(frame))
        i += 1
    return frames

root_path = "dataset"
data = {
    'paths': [],
    'labels': []
}
image_size = (224, 224)
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize(image_size, antialias=True),
])
num_frames = 48
if not os.path.exists('processed_data'):
    os.mkdir('processed_data')
for dir in os.listdir(root_path):
    video_folder = os.path.join(root_path, dir)
    if not os.path.exists(f'processed_data/{dir}'):
        os.mkdir(f'processed_data/{dir}')
    for idx, path in enumerate(os.listdir(video_folder)):
        video_path = os.path.join(video_folder, path)
        data['paths'].append(video_path)
        data['labels'].append(dir)
        frames = get_frames(video_path, num_frames)
        tensors = []
        for frame in frames:
            tensor = transform(frame)
            tensor = torch.unsqueeze(tensor, dim=0)
            tensors.append(tensor)
        input_video = torch.concatenate(tensors, dim=0)
        input_video = input_video.transpose(0, 1)
        torch.save(input_video, f'processed_data/{dir}/video{idx}.pt')