import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_module import *
from .mask_regression import *
from model import ResBlockG


class ResnetGenerator64(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator64, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 128)
        self.z_dim = z_dim
        num_w = z_dim + self.label_embedding.embedding_dim
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*16*ch))

        self.res2 = ResBlockG(ch*16, ch*8, upsample=True, num_w=num_w)
        self.res3 = ResBlockG(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res4 = ResBlockG(ch*4, ch*2, upsample=True, num_w=num_w)
        self.res5 = ResBlockG(ch*2, ch*1, upsample=True, num_w=num_w)
        self.final = nn.Sequential(nn.BatchNorm2d(ch),
                                   nn.LeakyReLU(0.01),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet(num_w+2, map_size=64)
        
        # self.self_attn = SelfAttn(num_w + 4)
        self.style_mapping = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim))
        )
        self.init_parameter()

        print(f"ResnetGenerator64 initialized")


    def forward(self, z, bbox, z_im=None, y=None, return_mask=False):
        b, o = z.size(0), z.size(1)
        z = z.cuda()
        bbox = bbox.cuda()
        
        label_embedding = self.label_embedding(y)

        # latent vector self-attention
        # mask_latent_vector = self.self_attn(torch.cat([label_embedding, z, bbox], dim=2)) # b*o*(num_w+4)
        mask_latent_vector = torch.cat([label_embedding, z, bbox[:,:,2:]], dim=2) # b*o*(num_w+2)
        if return_mask:
            mask, raw_mask = self.mask_regress(mask_latent_vector, bbox, return_raw=True)
        else:
            mask = self.mask_regress(mask_latent_vector, bbox)
        w = torch.cat( [label_embedding, self.style_mapping(z.view(b*o, -1)).view(b,o,-1)], dim=2)  # b*o*num_w
        
        if z_im is None:
            z_im = torch.randn((b, self.z_dim), device=z.device)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x = self.res2(x, w, mask)
        # 16x16
        x = self.res3(x, w, mask)
        # 32x32
        x = self.res4(x, w, mask)
        # 64x64
        x = self.res5(x, w, mask)
        x = self.final(x)
        return x if not return_mask else [x, mask, raw_mask]

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class ResnetGenerator128(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator128, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 160)
        self.z_dim = z_dim
        num_w = z_dim + self.label_embedding.embedding_dim
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*8*ch))

        self.res1 = ResBlockG(ch*8, ch*8, upsample=True, num_w=num_w)
        self.res2 = ResBlockG(ch*8, ch*4, upsample=True, num_w=num_w)
        self.res3 = ResBlockG(ch*4, ch*4, upsample=True, num_w=num_w)
        self.res4 = ResBlockG(ch*4, ch*2, upsample=True, num_w=num_w)
        self.res5 = ResBlockG(ch*2, ch*1, upsample=True, num_w=num_w)
        self.final = nn.Sequential(nn.BatchNorm2d(ch),
                                   nn.LeakyReLU(0.01),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())

        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet(num_w+2, map_size=128)

        self.style_mapping = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim))
        )
        self.init_parameter()
        print(f"ResnetGenerator128 initialized")

    def forward(self, z, bbox, z_im=None, y=None, return_mask=False):
        b, o = z.size(0), z.size(1)
        z, bbox = z.cuda(), bbox.cuda()
        
        label_embedding = self.label_embedding(y)

        mask_latent_vector = torch.cat([label_embedding, z, bbox[:,:,2:]], dim=2) # b*o*(num_w+2)
        if return_mask:
            mask, raw_mask = self.mask_regress(mask_latent_vector, bbox, return_raw=True)
        else:
            mask = self.mask_regress(mask_latent_vector, bbox)
        w = torch.cat( [label_embedding, self.style_mapping(z.view(b*o, -1)).view(b,o,-1)], dim=2)  # b*o*num_w
        
        if z_im is None:
            z_im = torch.randn((b, self.z_dim), device=z.device)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 8x8
        x = self.res1(x, w, mask)
        # 16x16
        x = self.res2(x, w, mask)
        # 32x32
        x = self.res3(x, w, mask)
        # 64x64
        x = self.res4(x, w, mask)
        # 128x128
        x = self.res5(x, w, mask)
        # to RGB
        x = self.final(x)
        return x if not return_mask else [x, mask, raw_mask]

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)



class ResnetGenerator256(nn.Module):
    def __init__(self, ch=64, z_dim=128, num_classes=10, output_dim=3):
        super(ResnetGenerator256, self).__init__()
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, 192)
        self.z_dim = z_dim
        num_w = z_dim + self.label_embedding.embedding_dim
        
        channels = [16, 8, 8, 4, 4, 2, 1]
        self.fc = nn.utils.spectral_norm(nn.Linear(z_dim, 4*4*channels[0]*ch))

        res = []
        for channel_in, channel_out in zip(channels[:-1], channels[1:]):
            res.append(ByPassFilter(
                ResBlockG(ch*channel_in, ch*channel_out, upsample=True, num_w=num_w)))
        self.res = nn.Sequential(*res)
        
        self.final = nn.Sequential(nn.BatchNorm2d(ch),
                                   nn.LeakyReLU(0.01),
                                   conv2d(ch, output_dim, 3, 1, 1),
                                   nn.Tanh())
        
        
        # mapping function
        mapping = list()
        self.mapping = nn.Sequential(*mapping)

        self.mask_regress = MaskRegressNet(num_w+2, map_size=256)

        self.style_mapping = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim)),
            nn.LeakyReLU(0.01),
            nn.utils.spectral_norm(nn.Linear(z_dim, z_dim))
        )
        self.init_parameter()
        print(f"ResnetGenerator256 initialized")

    def forward(self, z, bbox, z_im=None, y=None, return_mask=False):
        b, o = z.size(0), z.size(1)
        z, bbox = z.cuda(), bbox.cuda()
        
        label_embedding = self.label_embedding(y)

        mask_latent_vector = torch.cat([label_embedding, z, bbox[:,:,2:]], dim=2) # b*o*(num_w+2)
        if return_mask:
            mask, raw_mask = self.mask_regress(mask_latent_vector, bbox, return_raw=True)
        else:
            mask = self.mask_regress(mask_latent_vector, bbox)
        w = torch.cat( [label_embedding, self.style_mapping(z.view(b*o, -1)).view(b,o,-1)], dim=2)  # b*o*num_w
        
        if z_im is None:
            z_im = torch.randn((b, self.z_dim), device=z.device)

        # 4x4
        x = self.fc(z_im).view(b, -1, 4, 4)
        # 256x256
        x, w, mask = self.res([x, w, mask])
        # to RGB
        x = self.final(x)
        return x if not return_mask else [x, mask, raw_mask]

    def init_parameter(self):
        for k in self.named_parameters():
            if k[1].dim() > 1:
                torch.nn.init.orthogonal_(k[1])
            if k[0][-4:] == 'bias':
                torch.nn.init.constant_(k[1], 0)

class ByPassFilter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, args):
        x = self.model(*args)
        return [x] + args[1:] # These are w and masks
