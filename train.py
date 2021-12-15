import argparse
import datetime
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from data.cocostuff_loader import *
from data.vg import *
import data as data_util
from model.rcnn_discriminator import *
from model.resnet_generator import *
from model.sync_batchnorm import DataParallelWithCallback
from utils.logger import setup_logger
from utils.util import *

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def get_dataset(dataset, img_size):
    if dataset == "coco":
        data = CocoSceneGraphDataset(image_dir='./datasets/coco/train2017/',
                                        instances_json='./datasets/coco/annotations/instances_train2017.json',
                                        stuff_json='./datasets/coco/annotations/stuff_train2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=True)
    elif dataset == 'vg':
        with open("./datasets/vg/vocab.json", "r") as read_file:
            vocab = json.load(read_file)
        data = VgSceneGraphDataset(vocab=vocab, h5_path='./datasets/vg/train.h5',
                                      image_dir='./datasets/vg/images/',
                                      image_size=(img_size, img_size), max_objects=10, left_right_flip=True)
    return data

def keyword_dict(model, keyword): 
    return {name: param.mean() for name, param in model.named_parameters() if keyword in name}

def target_dict(model, target, keywords):  
    return {x:getattr(y, target) for x,y in model.named_modules() if any(x.endswith(k) for k in keywords) }  


def main(args):
    # parameters
    img_size = args.img_size
    assert img_size in [64, 128, 256, 512]
    z_dim = 128 # z_img
    lamb_obj = 1.0
    lamb_img = 0.1
    num_classes = 184 if args.dataset == 'coco' else 179
    
    # train D only in the first {DELAY_G_TRAIN} iters of each epoch
    DELAY_G_TRAIN = 30

    # data loader
    train_data = get_dataset(args.dataset, img_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, sampler=train_sampler, 
        drop_last=True, num_workers=1)
    val_dataset = data_util.get_dataset(args.dataset, img_size, left_right_flip=False, train=False) if dist.get_rank() == 0 else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, drop_last=False, shuffle=True, num_workers=1) if dist.get_rank()==0 else None

    # Load model
    netG = globals()[f'ResnetGenerator{img_size}'](num_classes=num_classes, output_dim=3).cuda()
    netD = globals()[f'CombineDiscriminator{img_size}'](num_classes=num_classes).cuda()

    # use DDP
    parallel = True
    if parallel:
        process_group = dist.new_group(list(range(dist.get_world_size())))
        if dist.get_world_size() > 1:
            netG = nn.SyncBatchNorm.convert_sync_batchnorm(netG, process_group)
        netG = DDP(netG, device_ids=[args.local_rank])
        netD = DDP(netD, device_ids=[args.local_rank])

    if dist.get_rank()==0:
        # to record hidden features
        G_recorder = HiddenFeatureRecoder(netG)
        D_recorder = HiddenFeatureRecoder(netD)

    
    g_lr, d_lr = args.g_lr, args.d_lr
    gen_parameters = []
    for key, value in dict(netG.named_parameters()).items():
        if value.requires_grad:
            if 'mapping' in key or "rho" in key:
                gen_parameters += [{'params': [value], 'lr': g_lr * 0.1}]
            else:
                gen_parameters += [{'params': [value], 'lr': g_lr}]

    # beta1 = 0 beta2=0.999
    g_optimizer = torch.optim.Adam(gen_parameters, betas=(0, 0.999))
    # use scheduler to reduce the learning rate to 0.1 & 0.01 in [milestones]
    milestones = [120, 160]
    g_scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, milestones)

    dis_parameters = []
    for key, value in dict(netD.named_parameters()).items():
        if value.requires_grad:
            if 'alpha' in key:
                dis_parameters += [{'params': [value], 'lr': d_lr * 0.2}]
            else:
                dis_parameters += [{'params': [value], 'lr': d_lr}]
    d_optimizer = torch.optim.Adam(dis_parameters, betas=(0, 0.999))
    d_scheduler = torch.optim.lr_scheduler.MultiStepLR(d_optimizer, milestones)

    if dist.get_rank()==0:
        # make dirs
        if not os.path.exists(args.out_path):
            print('mkdir args.out_path')
            os.makedirs(args.out_path)
        if not os.path.exists(os.path.join(args.out_path, 'model/')):
            os.mkdir(os.path.join(args.out_path, 'model/'))
        writer = SummaryWriter(os.path.join(args.out_path, 'log'))

        logger = setup_logger("LAMA", args.out_path, 0)
        logger.info(netG)
        logger.info(netD)
        
    g_loss, g_loss_fake, g_loss_obj, g_out_fake, g_out_obj = [torch.tensor(0.0)]*5

    start_time = time.time()
    epochs = trange(args.total_epoch) if dist.get_rank()==0 else range(args.total_epoch)
    for epoch in epochs:
        netG.train()
        netD.train()

        e_loader = enumerate(train_dataloader)
        e_loader = tqdm(e_loader, leave=False) if dist.get_rank()==0 else e_loader
        for idx, data in e_loader:
            real_images, label, bbox = data
            label, bbox = label.long().unsqueeze(-1), bbox.float()

            # update D network
            netD.zero_grad()
            d_out_real, d_out_robj = netD(real_images, bbox, label)
            d_loss_real = F.relu(1.0 - d_out_real).mean()
            d_loss_robj = F.relu(1.0 - d_out_robj).mean()

            z = torch.randn(real_images.size(0), bbox.size(1), z_dim)
            fake_images = netG(z, bbox, y=label.squeeze(dim=-1))
            d_out_fake, d_out_fobj = netD(fake_images.detach(), bbox, label)
            d_loss_fake = F.relu(1.0 + d_out_fake).mean()
            d_loss_fobj = F.relu(1.0 + d_out_fobj).mean()

            d_loss = lamb_obj * (d_loss_robj + d_loss_fobj) + lamb_img * (d_loss_real + d_loss_fake)
            d_loss.backward()
            d_optimizer.step()

            # update G network
            if (idx % 1) == 0 and idx > DELAY_G_TRAIN:
                netG.zero_grad()
                g_out_fake, g_out_obj = netD(fake_images, bbox, label)
                g_loss_fake = - g_out_fake.mean()
                g_loss_obj = - g_out_obj.mean()

                g_loss = g_loss_obj * lamb_obj + g_loss_fake * lamb_img
                g_loss.backward()
                g_optimizer.step()

            if dist.get_rank()==0:
                iterations = epoch * len(train_dataloader) + idx + 1
                if (idx % 10) == 0:
                    writer.add_scalar("D/d_loss", d_loss.item(), iterations)
                    writer.add_scalars("D/d_loss_realfake",
                        {"real": d_loss_real.item(), "fake": d_loss_fake.item()}, iterations)
                    writer.add_scalars("D/d_loss_obj",
                        {"real": d_loss_robj.item(), "fake": d_loss_fobj.item()}, iterations)
                    writer.add_scalars("D/d_out_realfake",
                        {"real": d_out_real.mean().item(), 
                        "fake": d_out_fake.mean().item(),
                        "gap": d_out_real.mean().item()-d_out_fake.mean().item()}, iterations)
                    writer.add_scalars("D/d_out_obj",
                        {"real": d_out_robj.mean().item(), 
                        "fake": d_out_fobj.mean().item(),
                        "gap": d_out_robj.mean().item()-d_out_fobj.mean().item()}, iterations)
                    writer.add_scalar("G/g_loss", g_loss.item(), iterations)
                    writer.add_scalar("G/g_loss_fake", g_loss_fake.item(), iterations)
                    writer.add_scalar("G/g_loss_obj", g_loss_obj.item(), iterations)
                    writer.add_scalar("G/g_out_fake", g_out_fake.mean().item(), iterations)
                    writer.add_scalar("G/g_out_obj", g_out_obj.mean().item(), iterations)
                    writer.add_scalars('G/rho', keyword_dict(netG, 'rho'), iterations)
                    
                # record hidden features with hooks    
                if idx == DELAY_G_TRAIN+1:
                    G_recorder.hook()
                    D_recorder.hook()
                elif idx == DELAY_G_TRAIN+2:
                    G_recorder.write(writer, "netG", epoch)
                    D_recorder.write(writer, "netD", epoch)
                    G_recorder.remove()
                    D_recorder.remove()

                if (idx+1) % 500 == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))
                    logger.info("Time Elapsed: [{}]".format(elapsed))
                    logger.info("Step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, g_out_fake: {:.4f} ".format(epoch + 1,
                                                                                                            idx + 1,
                                                                                                            d_loss_real.item(),
                                                                                                            d_loss_fake.item(),
                                                                                                            g_loss_fake.item()))
                    logger.info("             d_obj_real: {:.4f}, d_obj_fake: {:.4f}, g_obj_fake: {:.4f} ".format(
                                                                                                            d_loss_robj.item(),
                                                                                                            d_loss_fobj.item(),
                                                                                                            g_loss_obj.item()))

                    # record learning rate
                    writer.add_scalar("LR/d_lr", d_scheduler.get_last_lr()[0], iterations)
                    writer.add_scalar("LR/g_lr", g_scheduler.get_last_lr()[0], iterations)
                    # record images
                    writer.add_image("images/real images", make_grid(real_images.cpu().data * 0.5 + 0.5, nrow=4), iterations)
                    writer.add_image("images/fake images", make_grid(fake_images.cpu().data * 0.5 + 0.5, nrow=4), iterations)
                    # record epoch
                    writer.add_scalar("Time/epoch", epoch, iterations)
                    # report alpha and weights
                    writer.add_scalars(
                        'D/alpha', keyword_dict(netD, 'alpha'), iterations)
                    G_alpha = keyword_dict(netG, 'alpha')
                    writer.add_scalars('G/alpha', G_alpha, iterations)
                    writer.add_scalars('G/res_alpha', 
                        {k:v for k, v in G_alpha.items() if "module.res" in k}, iterations)
                    writer.add_scalars('G/mask_alpha', 
                        {k:v for k, v in G_alpha.items() if "module.mask" in k}, iterations)
                    writer.add_scalars('G/noise_weight_mean', 
                        {k:F.softplus(v) for k, v in keyword_dict(netG, 'noise_weight_seed').items()}, iterations)

            # end one epoch

        if dist.get_rank()==0:
            # record weights
            write_weights_grad(writer, netG, prefix='G', step=epoch)
            write_weights_grad(writer, netD, prefix='D', step=epoch)

            # record the avg training time of one epoch
            writer.add_scalar("Time/epoch_avg", (time.time()-start_time)/(epoch+1), epoch)

            out_real_val, out_robj_val = [], []
            # record output of D on validation dataset
            for idx, val_data in tqdm(enumerate(val_dataloader), desc='validation'):
                real_images, label, bbox = val_data
                with torch.no_grad():
                    d_out_real_val, d_out_robj_val = netD(real_images, bbox, label)
                out_real_val.append(d_out_real_val.detach().cpu())
                out_robj_val.append(d_out_robj_val.detach().cpu())
                if idx > 128:
                    break
            avg_real_val = torch.cat(out_real_val, dim=0)
            avg_robj_val = torch.cat(out_robj_val, dim=0)
            
            writer.add_scalars("D/d_img_comparison",
                        {"real_train": d_out_real.mean().item(), 
                        "fake_train":  d_out_fake.mean().item(),
                        "real_val":  avg_real_val.mean().item(),
                        }, epoch)
            writer.add_scalars("D/d_obj_comparison",
                        {"robj_train": d_out_robj.mean().item(),
                        "fobj_train":  d_out_fobj.mean().item(),
                        "robj_val":  avg_robj_val.mean().item()
                        }, epoch)
                    

            # save model
            if (epoch + 1) % 5 == 0:
                save_path = os.path.join(args.out_path, 'model/', 'G_%d.pth' % (epoch+1))
                torch.save(netG.state_dict(), save_path)
                    
        # use scheduler
        g_scheduler.step()
        d_scheduler.step()

        # end whole training
    if dist.get_rank()==0:    
        writer.flush()
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='mini-batch size of training data. Default: 128')
    parser.add_argument('--total_epoch', type=int, default=200,
                        help='number of total training epoch')
    parser.add_argument('--d_lr', type=float, default=0.0003,
                        help='learning rate for discriminator')
    parser.add_argument('--g_lr', type=float, default=0.0001,
                        help='learning rate for generator')
    parser.add_argument('--out_path', type=str, default='./outputs/',
                        help='path to output files')
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    parser.add_argument('--local_rank', default=-1, type=int, 
                        help='node rank for distributed training')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    main(args)
