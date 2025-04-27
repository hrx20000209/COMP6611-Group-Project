import torch
import torch.nn as nn
import torch.nn.functional as F

class RadarEncoder(nn.Module):
    def __init__(self, input_dim=5, feature_dim=128):
        super(RadarEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T, N, 5)
        B, T, N, _ = x.shape
        x = x.view(B * T, N, -1)  # (B*T, N, 5)
        x = self.mlp(x)  # (B*T, N, feature_dim)
        x = torch.max(x, dim=1)[0]  # (B*T, feature_dim) max pooling over points
        x = x.view(B, T, -1)  # (B, T, feature_dim)
        return x

class VideoEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(VideoEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.cnn(x)  # (B*T, 128, 1, 1)
        x = x.view(B*T, 128)
        x = self.fc(x)  # (B*T, feature_dim)
        x = x.view(B, T, -1)  # (B, T, feature_dim)
        return x

class TemporalFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=4):
        super(TemporalFusion, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, input_dim)
        out, _ = self.gru(x)  # (B, T, hidden_dim)
        out = out[:, -1, :]  # 只取最后一个 timestep 的输出
        out = self.classifier(out)
        return out

class MultiModalActionModel(nn.Module):
    def __init__(self, radar_feature_dim=128, video_feature_dim=128, hidden_dim=128, num_classes=4):
        super(MultiModalActionModel, self).__init__()
        self.radar_encoder = RadarEncoder(input_dim=5, feature_dim=radar_feature_dim)
        self.video_encoder = VideoEncoder(feature_dim=video_feature_dim)
        self.temporal_fusion = TemporalFusion(
            input_dim=radar_feature_dim + video_feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )

    def forward(self, radar_input, video_input):
        radar_feat = self.radar_encoder(radar_input)  # (B, T, radar_feature_dim)
        video_feat = self.video_encoder(video_input)  # (B, T, video_feature_dim)
        fused = torch.cat([radar_feat, video_feat], dim=-1)  # (B, T, radar+video)
        output = self.temporal_fusion(fused)  # (B, num_classes)
        return output
