import torch.nn as nn
import torch

import modules.functional as F
from modules.voxelization import Voxelization
from modules.util import SharedMLP,Point_Branch_TF
from modules.se import SE3d

class CCMNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_sample, radius, num_neighbors, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),  
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_trans = Point_Branch_TF(in_channels, out_channels, num_sample, radius, num_neighbors)
        self.point_features = SharedMLP(in_channels, out_channels)
        self.conv2 = nn.Conv1d(2 * out_channels, out_channels, 1)
        

    def forward(self, inputs):
        features, coords = inputs
        # voxel_features：bcrrr   voxel_coords: b 3 n
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features) # b c r r r
        f1 = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        f2 = self.point_trans(features, coords)
        f3 = self.point_features(features)
        
        fused_features = self.conv2(torch.cat((f1, f2), 1)) + f3
        return fused_features, coords
