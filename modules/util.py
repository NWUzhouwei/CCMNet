from statistics import mean
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from modules.ball_query import BallQuery
import modules.functional as ff

class Point_Branch_TF(nn.Module): 
    def __init__(self, in_channels, out_channels, num_sample, radius, num_neighbors):
        super().__init__()
        self.out_channels = out_channels
        self.num_sample = num_sample # m
        self.raidus = radius # r
        self.num_neighbors = num_neighbors # k

        self.ballquery = BallQuery(radius, num_neighbors, include_coordinates = False)
        
        # LinearAttention
        self.lin_q = nn.Linear(2*in_channels, out_channels)
        self.lin_k = nn.Linear(2*in_channels, out_channels)
        self.lin_v = nn.Linear(2*in_channels, out_channels)
        self.trans_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.after_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        # middle layer
        self.middle = nn.Linear(2 * out_channels, out_channels)
        self.middle_norm = nn.BatchNorm1d(out_channels)
        self.middle_act = nn.ReLU()
       
        # cross attention
        self.num_heads = 8
        self.scale = (out_channels // self.num_heads) ** -0.5
        self.q_map = nn.Linear(in_channels, out_channels)
        self.k_map = nn.Linear(out_channels, out_channels)
        self.v_map = nn.Linear(out_channels, out_channels)
        self.attn_drop = nn.Dropout(0.5)
        self.proj = nn.Linear(out_channels, out_channels)
        
    def forward(self, features, coords):
        
        # FPS get center point
        # b 3 m    b in m
        centers_coords , centers_feature= ff.furthest_point_sample(coords, features, self.num_sample) # b 3 m 中心点
        b, c, m = centers_feature.shape
        # ball query  b c m k
        neighbor_features = self.ballquery(coords, centers_coords, features) # b c m k
        
        x = centers_feature.view(b, c, m, 1).repeat(1, 1, 1, self.num_neighbors)# b c m k
      
        # b m k 2c
        edge_features = torch.cat((neighbor_features-x, x), dim=1).permute(0, 2, 3, 1)

        # inter self-attention
        q = self.lin_q(edge_features) # bmkc
        k = self.lin_k(edge_features).permute(0, 1, 3, 2) # bmck
        v = self.lin_v(edge_features).permute(0, 1, 3, 2) # bmck
        energy = torch.matmul(q, k) # bmkk
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=2, keepdim=True))
        x_r = torch.matmul(v, attention).permute(0, 2, 1, 3) # bmck->bcmk
        x_r = self.act(self.after_norm(self.trans_conv(x_r))) # bcmk

        # middle layer
        x_max = x_r.max(dim=-1, keepdim=False)[0] # max-pooling bcm
        x_mean = x_r.mean(dim=-1, keepdim = False) # avg-pooling bcm
        x_r = torch.cat((x_max, x_mean), 1) # concat    b 2c m
        x_r = self.middle(x_r.permute(0, 2, 1))
        x_r = self.middle_act(self.middle_norm(x_r.permute(0, 2, 1))) # bcn


        # outer cross-attention
        q = self.q_map(centers_feature.permute(0, 2, 1)).view(b, m, self.num_heads, self.out_channels // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(x_r.permute(0, 2, 1)).view(b, m, self.num_heads, self.out_channels // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(x_r.permute(0, 2, 1)).view(b, m, self.num_heads, self.out_channels // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_c = (attn @ v).transpose(1, 2).reshape(b, m, self.out_channels)
        x_c = self.proj(x_c) # bmc       
       
        # interpolate
        point_feature = ff.nearest_neighbor_interpolate(coords, centers_coords, x_c.permute(0, 2, 1)) # cross_attention
        return point_feature # bcn


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if dim == 1:
            conv = nn.Conv1d
            bn = nn.BatchNorm1d
        elif dim == 2:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
        else:
            raise ValueError
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        
        if dim != -1:
            layers = []
            for oc in out_channels:
                layers.extend([
                    conv(in_channels, oc, 1),
                    bn(oc),
                    nn.ReLU(True),
                ])
                in_channels = oc
            self.layers = nn.Sequential(*layers)
        else:
            self.layers = nn.Sequential(*ll)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return (self.layers(inputs[0]), *inputs[1:])
        else:
            return self.layers(inputs)