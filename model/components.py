from torch import nn
from torch.nn import functional as F
from .norm_module import *

# adopted from https://github.com/rosinality/stylegan2-pytorch/blob/master/model.py#L280
class NoiseInjection(nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.noise_weight_seed = nn.Parameter(torch.tensor(0.0))
        self.full = full

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, [1, channel][self.full], height, width).normal_()
        return image + F.softplus(self.noise_weight_seed) * noise

class ChannelwiseNoiseInjection(nn.Module):
    def __init__(self, num_channels, full=False):
        super().__init__()
        self.noise_weight_seed = nn.Parameter(torch.zeros((1, num_channels, 1, 1)))
        self.num_channels = num_channels
        self.full = full

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, [1, channel][self.full], height, width).normal_()
        return image + F.softplus(self.noise_weight_seed) * noise
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_channels) + ')'

class ResBlockG(nn.Module):
    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1, upsample=False, num_w=128):
        super().__init__()
        self.upsample = upsample
        self.h_ch = h_ch if h_ch else out_ch
        self.conv1 = conv2d(in_ch, self.h_ch, ksize, pad=pad, bias=False)
        self.conv2 = conv2d(self.h_ch, out_ch, ksize, pad=pad, bias=False)
        self.b1 = SpatialAdaptiveSynBatchGroupNorm2d(in_ch, num_w=num_w)
        self.b2 = SpatialAdaptiveSynBatchGroupNorm2d(self.h_ch, num_w=num_w)
        self.learnable_sc = in_ch != out_ch or upsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.activation = nn.LeakyReLU(0.01)

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.out_ch = out_ch
        # self.noise1 = NoiseInjection()
        # self.noise2 = NoiseInjection()
        self.noise1 = ChannelwiseNoiseInjection(self.h_ch)
        self.noise2 = ChannelwiseNoiseInjection(out_ch)
        
    def residual(self, in_feat, w, bbox):
        x = in_feat
        x = self.b1(x, w, bbox)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.noise1(x)
        x = self.b2(x, w, bbox)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.noise2(x)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            if self.upsample:
                x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.c_sc(x)
        return x

    def forward(self, in_feat, w, bbox):
        return self.alpha * self.residual(in_feat, w, bbox) + self.shortcut(in_feat)



class ResBlockD(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, downsample=False):
        super().__init__()
        self.conv1 = conv2d(in_ch, out_ch, ksize, 1, pad)
        self.conv2 = conv2d(out_ch, out_ch, ksize, 1, pad)
        self.activation = nn.LeakyReLU(0.01)
        self.downsample = downsample
        self.learnable_sc = (in_ch != out_ch) or downsample
        if self.learnable_sc:
            self.c_sc = conv2d(in_ch, out_ch, 1, 1, 0)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def residual(self, in_feat):
        x = in_feat
        x = self.conv1(self.activation(x))
        x = self.conv2(self.activation(x))
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                x = F.avg_pool2d(x, 2)
        return x

    def forward(self, in_feat):
        return self.alpha.clamp(-1,1) * self.residual(in_feat) + self.shortcut(in_feat)

def conv2d(in_feat, out_feat, kernel_size=3, stride=1, pad=1, spectral_norm=True, bias=True):
    conv = nn.Conv2d(in_feat, out_feat, kernel_size, stride, pad, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(conv, eps=1e-4)
    else:
        return conv


class MaskRegressBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, bias = False):
        super().__init__()
        conv = list()
        conv.append(nn.BatchNorm2d(channels))
        conv.append(nn.LeakyReLU(0.01))
        conv.append(conv2d(channels, channels, kernel_size, bias = bias))
        self.conv = nn.Sequential(*conv)
        self.alpha = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        return x + self.alpha * self.conv(x)


# BGN+SPADE 
class SpatialAdaptiveSynBatchGroupNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512):
        super().__init__()
        self.num_features = num_features
        self.weight_proj = nn.utils.spectral_norm(
            nn.Linear(num_w, num_features))
        self.bias_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.batch_norm2d = nn.BatchNorm2d(num_features, eps=1e-5, affine=False,
                            momentum=0.1, track_running_stats=True)

        self.group_norm = nn.GroupNorm(4, num_features, eps=1e-5, affine=False)
        self.rho = nn.Parameter(torch.tensor(0.1)) # the ratio of GN

        self.alpha = nn.Parameter(torch.tensor(0.0)) # the scale of the affined 

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        self.batch_norm2d._check_input_dim(x)
        # use BGN
        output_b = self.batch_norm2d(x)
        output_g = self.group_norm(x)
        output = output_b + self.rho.clamp(0,1) * (output_g - output_b)

        b, o, _, _ = bbox.size()
        _, _, h, w = x.size()
        bbox = F.interpolate(bbox, size=(h, w), mode='bilinear', align_corners=False) # b o h w
        weight, bias = self.weight_proj(vector), self.bias_proj(vector) # b*o d

        bbox_non_spatial = bbox.view(b, o, -1) # b o h*w
        bbox_non_spatial_margin = bbox_non_spatial.sum(dim=1, keepdim=True) + torch.tensor(1e-4) # b 1 h*w
        bbox_non_spatial.div_(bbox_non_spatial_margin)
        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1) # b o d
        weight.transpose_(1, 2), bias.transpose_(1, 2) # b d o
        weight, bias = torch.bmm(weight, bbox_non_spatial), torch.bmm(bias, bbox_non_spatial) # b d h*w
        # weight.div_(bbox_non_spatial_margin), bias.div_(bbox_non_spatial_margin) # b d h*w
        weight, bias = weight.view(b, -1, h, w), bias.view(b, -1, h, w)

        # weight = torch.sum(bbox * weight, dim=1, keepdim=False) / \
        #     (torch.sum(bbox, dim=1, keepdim=False) + 1e-6) # b d h w
        # bias = torch.sum(bbox * bias, dim=1, keepdim=False) / \
        #     (torch.sum(bbox, dim=1, keepdim=False) + 1e-6) # b d h w
        affined = weight * output + bias
        return output + self.alpha.clamp(-1, 1) * affined

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
