from model import MaskRegressBlock, conv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bilinear import *
# from .sync_batchnorm import SynchronizedBatchNorm2d
from .norm_module import AdaptiveBatchNorm2d
import pickle
import numpy as np
from torch.nn import Parameter


class MaskRegressNet(nn.Module):
    def __init__(self, obj_feat=128, mask_size=32, map_size=64):
        super().__init__()
        self.mask_size = mask_size
        self.map_size = map_size
        self.hidden_feat, hidden_feat = 128, 128

        self.fc = nn.utils.spectral_norm(
            nn.Linear(obj_feat, hidden_feat * 4 * 4))

        self.conv1 = MaskRegressBlock(hidden_feat)
        self.conv2 = MaskRegressBlock(hidden_feat)
        self.conv3 = MaskRegressBlock(hidden_feat)
        # self.conv4 = MaskRegressBlock(hidden_feat)
        
        final = list()
        final.append(nn.BatchNorm2d(hidden_feat))
        final.append(nn.LeakyReLU(0.01))
        final.append(nn.utils.spectral_norm(nn.Conv2d(hidden_feat, 1, 1, 1)))
        final.append(nn.Sigmoid())
        self.final = nn.Sequential(*final)

        match_dim = obj_feat // 4
        self.query_affine = nn.utils.spectral_norm(nn.Linear(obj_feat, match_dim))
        self.key_affine = nn.utils.spectral_norm(nn.Linear(obj_feat, match_dim))
        self.query_local = nn.Sequential(
            MaskAdjustBlock(match_dim),
            MaskAdjustBlock(match_dim)
        )
        self.mask_adjust_alpha = nn.Parameter(torch.tensor(0.0))

        self.init_parameter()

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

    def forward(self, obj_feat, bbox, return_raw=False):
        """
        :param obj_feat: (b*num_o, feat_dim)
        :param bbox: (b, num_o, 4)
        :return: bbmap: (b, num_o, map_size, map_size)
        """
        b, num_o, _ = bbox.size()
        obj_feat = obj_feat.view(b * num_o, -1)
        x = self.fc(obj_feat) # 4 * 4
        x = self.conv1(x.view(b * num_o, self.hidden_feat, 4, 4))  # 4 * 4
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)     # 8 * 8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)     # 16 * 16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # x = self.conv4(x)     # 32 * 32
        x = self.final(x)     # 32 * 32 
        x = x.view(b, num_o, self.mask_size, self.mask_size)

        
        bbmap = masks_to_layout(bbox, x, self.map_size).view(b, num_o, self.map_size, self.map_size) 
        
        # the mask refined stage
        key = self.key_affine(obj_feat).view(b, num_o, -1) # b o d
        label_query = self.query_affine(obj_feat).view(b, num_o, -1).permute(0,2,1) # b d o
        bbmap_resized = F.interpolate(bbmap, size=(64,64), mode='bilinear', align_corners=False)
        bbmap_resized = bbmap_resized.view(b, num_o, 64*64) # b o w^2
        pixel_query = torch.bmm(label_query, bbmap_resized).view(b, -1, 64, 64) # b d w w
        local_query = self.query_local(pixel_query).view(b,-1,64**2) # b d w^2
        energy = torch.bmm(key, local_query).view(b, num_o, 64, 64) # b o w w
        # Tanh( f(bbmap)*alpha ) + 1 
        adjust = torch.tanh(energy * self.mask_adjust_alpha.clamp(-1, 1) ) + 1
        adjust = F.interpolate(adjust, size=(self.map_size, self.map_size), mode='bilinear', align_corners=False)

        bbmap_adjust = bbmap * adjust
        
        return bbmap_adjust if not return_raw else [bbmap_adjust, bbmap] 

# Adjust block, use GN to avoid perturbation
class MaskAdjustBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        conv = list()
        conv.append(nn.GroupNorm(4, channels))
        conv.append(nn.LeakyReLU(0.01))
        conv.append(
            nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1, bias=False))
            )
        self.conv = nn.Sequential(*conv)
        self.alpha = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        return x + self.alpha * self.conv(x)