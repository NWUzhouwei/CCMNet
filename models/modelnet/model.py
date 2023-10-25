import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import create_mlp_components, create_ccm_components

__all__ = ['Model']

class Model(nn.Module):
    
    blocks = ((64, 1, 32, 100, 0.03, 45), (128, 1, 16, 800, 0.01, 6), (64, 1, 32, 100, 0.03, 45), (128, 1, 16, 800, 0.01, 6), (64, 1, 32, 100, 0.03, 45)) # [1024]
    
    def __init__(self, num_classes=40, extra_feature_channels=3,
                 width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.in_channels = extra_feature_channels + 3 # 输入通道=6

        layers1, channels_point, concat_channels_point = create_ccm_components(
            blocks=self.blocks, in_channels=self.in_channels, with_se=True, normalize=False,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
            )
        self.point_features1 = nn.ModuleList(layers1)

        layers2, channels_cloud = create_mlp_components(
            in_channels=concat_channels_point, out_channels=[1024], # 可添加
            classifier=False, dim=2, width_multiplier=width_multiplier)
        self.point_features2 = nn.Sequential(*layers2)

        self.mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3), # 0.3
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 2,1)
        # inputs : [B, in_channels + S, N]
        features = inputs[:, :self.in_channels, :]

        coords = features[:, :3, :]
        out_features_list = []
        for i in range(len(self.point_features1)):
            features, _ = self.point_features1[i]((features, coords)) # b c n
            out_features_list.append(features)
        x = torch.cat(out_features_list, dim=1)
        x = self.point_features2(x)
        x = self.mlp(x.max(dim=-1, keepdim=False)[0]) # b n 
        x = F.log_softmax(x, dim=1)
        return x
