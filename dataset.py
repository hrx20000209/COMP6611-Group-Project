import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random


class MultiModalActionDataset(Dataset):
    def __init__(self, sample_list, label_map, radar_max_points=32, transform=None):
        """
        sample_list: list of (npz_path, label_name)
        label_map: dict {label_name: label_index}
        radar_max_points: 最大保留雷达点数（不足补零）
        transform: 对视频帧的transform（如resize）
        """
        self.samples = sample_list
        self.label_map = label_map
        self.radar_max_points = radar_max_points
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, label_name = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)

        # 处理雷达数据
        radar_sequence = data['radar']  # (T, N, 5)
        radar_tensor = self._process_radar(radar_sequence)

        # 处理视频数据
        video_sequence = data['video']  # (T, H, W, C)
        video_tensor = self._process_video(video_sequence)

        label = self.label_map[label_name]

        return radar_tensor, video_tensor, label

    def _process_radar(self, radar_sequence):
        """
        radar_sequence: (T, N, 5)
        保留每帧前 radar_max_points 个点，不足补零
        """
        frames = []
        for frame in radar_sequence:
            if frame.shape[0] >= self.radar_max_points:
                selected = frame[:self.radar_max_points, :]
            else:
                pad = np.zeros((self.radar_max_points - frame.shape[0], 5))
                selected = np.vstack((frame, pad))
            frames.append(selected)

        frames = np.stack(frames)  # (T, radar_max_points, 5)
        return torch.from_numpy(frames).float()

    def _process_video(self, video_sequence):
        """
        video_sequence: (T, H, W, C)
        """
        frames = []
        for frame in video_sequence:
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)

        frames = torch.stack(frames)  # (T, C, H, W)
        return frames


def build_dataset(root_dir, train_ratio=0.8, radar_max_points=32, transform=None):
    label_names = sorted(os.listdir(root_dir))  # 例如 ['sitting', 'squatting', 'standing', 'walking']
    label_map = {name: idx for idx, name in enumerate(label_names)}

    samples = []
    for label_name in label_names:
        label_dir = os.path.join(root_dir, label_name)
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.npz'):
                file_path = os.path.join(label_dir, file_name)
                samples.append((file_path, label_name))

    # 打乱并划分
    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_set = MultiModalActionDataset(train_samples, label_map, radar_max_points, transform)
    val_set = MultiModalActionDataset(val_samples, label_map, radar_max_points, transform)

    return train_set, val_set, label_map
