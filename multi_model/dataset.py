import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class MultiModalActionDataset(Dataset):
    def __init__(self, sample_list, label_map, radar_max_points=32, target_len=15, transform=None, radar_augment=False):
        """
        sample_list: list of (npz_path, label_name)
        label_map: dict {label_name: label_index}
        radar_max_points: 最大保留雷达点数（不足补零）
        target_len: 补齐或截断到固定帧数，比如15帧
        transform: torchvision transform，用于视频数据
        radar_augment: 是否对雷达数据做增强（只在训练集打开）
        """
        self.samples = sample_list
        self.label_map = label_map
        self.radar_max_points = radar_max_points
        self.target_len = target_len
        self.transform = transform
        self.radar_augment = radar_augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, label_name = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)

        radar_sequence = data['radar']  # List of dicts
        video_sequence = data['video']  # List of dicts

        radar_tensor = self._process_radar(radar_sequence)
        video_tensor = self._process_video(video_sequence)

        label = self.label_map[label_name]

        return radar_tensor, video_tensor, label

    def _process_radar(self, radar_sequence):
        frames = []
        for frame in radar_sequence:
            x = np.array(frame['x'])
            y = np.array(frame['y'])
            z = np.array(frame['z'])
            doppler = np.array(frame['doppler'])
            range_ = np.array(frame['range'])

            points = np.stack([x, y, z, doppler, range_], axis=-1)

            if self.radar_augment:
                points = self._augment_radar(points)

            # 简单归一化
            points[:, :3] /= 128.0  # x,y,z
            points[:, 3] /= 8208.0  # doppler
            points[:, 4] /= 5.0     # range

            if points.shape[0] >= self.radar_max_points:
                selected = points[:self.radar_max_points, :]
            else:
                pad = np.zeros((self.radar_max_points - points.shape[0], 5))
                selected = np.vstack((points, pad))

            frames.append(selected)

        frames = self._pad_or_truncate(frames, self.target_len)
        frames = np.stack(frames)
        return torch.from_numpy(frames).float()

    def _augment_radar(self, points):
        # 随机丢点
        keep_ratio = np.random.uniform(0.8, 1.0)
        num_keep = int(points.shape[0] * keep_ratio)
        indices = np.random.choice(points.shape[0], num_keep, replace=False)
        points = points[indices]

        # 小尺度噪声
        noise = np.random.normal(0, 0.01, size=points.shape)
        points += noise

        return points

    def _process_video(self, video_sequence):
        frames = []
        for frame in video_sequence:
            img = frame['frame']
            img = img[..., [2, 1, 0]]  # BGR -> RGB

            img = img.astype(np.uint8)

            if self.transform:
                img = self.transform(img)
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            frames.append(img)

        frames = self._pad_or_truncate(frames, self.target_len)
        frames = torch.stack(frames)
        return frames

    def _pad_or_truncate(self, frames, target_len):
        if len(frames) >= target_len:
            frames = frames[:target_len]
        else:
            pad = [frames[-1]] * (target_len - len(frames))
            frames = frames + pad
        return frames

def build_dataset(root_dir, train_ratio=0.8, radar_max_points=32, target_len=15, train_transform=None, val_transform=None):
    label_names = sorted(os.listdir(root_dir))
    label_map = {name: idx for idx, name in enumerate(label_names)}

    samples = []
    for label_name in label_names:
        label_dir = os.path.join(root_dir, label_name)
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.npz'):
                file_path = os.path.join(label_dir, file_name)
                samples.append((file_path, label_name))

    random.shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_set = MultiModalActionDataset(
        train_samples, label_map,
        radar_max_points=radar_max_points,
        target_len=target_len,
        transform=train_transform,
        radar_augment=True
    )

    val_set = MultiModalActionDataset(
        val_samples, label_map,
        radar_max_points=radar_max_points,
        target_len=target_len,
        transform=val_transform,
        radar_augment=False
    )

    return train_set, val_set, label_map
