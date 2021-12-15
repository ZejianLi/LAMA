import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from model import ResBlockG, ResBlockD, MaskRegressNet

def crop_resize(image, bbox, imsize=64, cropsize=28, label=None):
    """"
    :param image: (b, 3, h, w)
    :param bbox: (b, o, 4)
    :param imsize: input image size
    :param cropsize: image size after crop
    :param label:
    :return: crop_images: (b*o, 3, h, w)
    """
    crop_images = list()
    b, o, _ = bbox.size()
    if label is not None:
        rlabel = list()
    for idx in range(b):
        for odx in range(o):
            if torch.min(bbox[idx, odx]) < 0:
                continue
            crop_image = image[idx:idx+1, :, int(imsize*bbox[idx, odx, 1]):int(imsize*(bbox[idx, odx, 1]+bbox[idx, odx, 3])),
                               int(imsize*bbox[idx, odx, 0]):int(imsize*(bbox[idx, odx, 0]+bbox[idx, odx, 2]))]
            crop_image = F.interpolate(crop_image, size=(cropsize, cropsize), mode='bilinear', align_corners=False)
            crop_images.append(crop_image)
            if label is not None:
                rlabel.append(label[idx, odx, :].unsqueeze(0))
    # print(rlabel)
    if label is not None:
        #if len(rlabel) % 2 == 1:
        #    return torch.cat(crop_images[:-1], dim=0), torch.cat(rlabel[:-1], dim=0)
        return torch.cat(crop_images, dim=0), torch.cat(rlabel, dim=0)
    return torch.cat(crop_images, dim=0)


def truncted_random(num_o=8, thres=1.0, dim=128):
    z = torch.randn(1, num_o, dim)
    truncation_flag = (z.abs()>thres).float()
    return truncation_flag * torch.randn(1, num_o, dim) + (1.-truncation_flag) * z


def write_weights_grad(writer, model, prefix=None, step=0):
    """
    record the parameters and gradient in [model] in the [writer].
    """
    prefix_weight = '' if prefix is None else prefix+'/'
    prefix_grad = 'grad' if prefix is None else prefix+'_grad/'
    for name, param in model.named_parameters():
        writer.add_histogram(f'{prefix_weight}{name}', param.clone().cpu(), step)
        # writer.add_histogram(f'{prefix_grad}{name}',
        #                      param.grad.clone().cpu(), step)

class HiddenFeatureRecoder():
    """
    record the outputs of [model]'s children modules in [writer].
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.hiddenFeature = dict()
        self.handles = dict()
        self.filter_list = [ResBlockG, ResBlockD, MaskRegressNet]

    def hook(self):    
        for name, layer in self.model.named_modules():
            if any(isinstance(layer, l) for l in self.filter_list):
                # hook = lambda module, module_in, module_out: self.hiddenFeature.setdefault(name, module_out.clone().detach().cpu());module_out
                def this_hook(module, module_in, module_out, this_name=name):
                    self.hiddenFeature.setdefault(this_name, module_out.clone().detach().cpu())
                    return module_out
                self.handles[name] = layer.register_forward_hook(this_hook)
    
    def remove(self):
        for v in self.handles.values():
            v.remove()

    def write(self, writer, prefix=None, step=0):
        prefix = '' if prefix is None else prefix+'/'
        for k, v in self.hiddenFeature.items():
            v = v[0].detach().cpu()
            if v.dim()==3:
                v = v.unsqueeze(1)
            writer.add_image(prefix+k, make_grid(v, nrow=8), step)
        self.hiddenFeature = dict()