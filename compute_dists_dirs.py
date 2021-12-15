import argparse
import os
import DSmodels as models
import torch
from tqdm import tqdm
from torch import nn

class DSscore(nn.Module):
    def __init__(self, use_gpu=True):
        super().__init__()
        model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, version="0.1")
        self.model = model.cuda() if use_gpu else model
        self.results = []

    def forward(self, img_generated, img_real):
        img_generated = img_generated.detach()
        img_real = img_real.type_as(img_generated)
        with torch.no_grad():
            result = self.model.forward(img_generated, img_real)
        self.results.append(result)
        return result

    def mean_std(self):
        result = torch.cat(self.results, dim=0)
        mean = result.mean().item()
        std = result.std().item()
        self.results = []
        return mean, std

if __name__ == "__main__":
    d = DSscore()
    for _ in range(3):
        d(torch.randn(2,3,128,128).cuda(), torch.randn(2,3,128,128).cuda())
    print(d.avg())
    d = DSscore(False)
    for _ in range(3):
        d(torch.randn(2,3,128,128), torch.randn(2,3,128,128))
    print(d.avg())
    