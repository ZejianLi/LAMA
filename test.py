import argparse
from collections import OrderedDict

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm, trange
import seaborn as sns

from compute_dists_dirs import DSscore
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator import *
from utils.util import *
from data import get_dataset

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # parameters
    img_size = args.img_size
    assert img_size in [64, 128, 256, 512]

    num_classes = 184 if 'coco' in args.dataset.lower()  else 179

    dataset = get_dataset(args.dataset, img_size, left_right_flip=False, train=args.train)
    if args.image_id_savepath is not None and 'coco' in args.dataset.lower():
        np.savetxt(args.image_id_savepath, np.array(dataset.image_ids).astype(int))
        print(f"Image ids of generated images in COCO are save in {args.image_id_savepath}.")
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1,
                    drop_last=False, shuffle=False, num_workers=1)


    netG = globals()[f'ResnetGenerator{img_size}'](
        num_classes=num_classes, output_dim=3).cuda()

    print(f'Use ResnetGenerator{img_size} in {args.model_path}')

    if not os.path.isfile(args.model_path):
        print('I fail to find the pretrained model.')
        return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    thres = args.truncation_threshold
    repeat = args.repeat
    save_path = os.path.join(args.sample_path, args.dataset) + f"{img_size}_repeat{repeat}_thres{thres:.1f}"
    print("Saved in {}".format(save_path) if not args.NOSaving else "")

    if args.cropped_size > 0:
        save_path += f"_cropped_{args.cropped_size}/"
        for idx in range(num_classes):
            os.makedirs(os.path.join(save_path, str(idx)), exist_ok = True)
    else:
        os.makedirs(save_path, exist_ok=True)
    DSflag = args.DS
    DS = DSscore() if DSflag else None

    for idx, data in tqdm(enumerate(dataloader)):
        real_image, label, bbox = data
        real_image, label = real_image.cuda(), label.long().unsqueeze(-1)
        fake_images = []
        for r in range(repeat):
            z_obj = truncted_random(num_o=bbox.size(1), thres=thres).cuda()
            z_im = truncted_random(num_o=1, thres=thres).view(1, -1).cuda()
            with torch.no_grad():
                fake_image = netG.forward(z_obj, bbox.float().cuda(), z_im, label.squeeze(dim=-1).cuda())
                fake_images.append(fake_image)
            fake_arrays = fake_image[0].cpu().detach().mul(0.5).add(0.5).mul(255).to(torch.uint8).numpy().transpose(1, 2, 0)
            if not args.NOSaving:
                imageio.imwrite(f"{save_path}/sample_{idx}_{r}.png", fake_arrays)

            # extract image crops
            if args.cropped_size > 0:
                fakeImage = Image.fromarray(fake_arrays)
                bbox_rec = bbox.clone()
                bbox_rec[:, :, 2] = bbox[:, :, 2] + bbox[:, :, 0]  # x2 = x1 + w
                bbox_rec[:, :, 3] = bbox[:, :, 3] + bbox[:, :, 1]  # y2 = y1 + h
                bbox_rec = bbox_rec.mul(real_image.size(2)).to(torch.int)  # the position
                for idx1, bl in enumerate(zip(bbox_rec[0], label.flatten().tolist())):
                    b, l = bl[0].tolist(), bl[1]
                    if l > 0: # remove background
                        cropped = fakeImage.crop(b).resize( (args.cropped_size, args.cropped_size), resample=Image.BICUBIC )
                        cropped.save(os.path.join(save_path, str(l), f"{idx}_{r}_{idx1}.png"))
        DS(fake_images[0], fake_images[1]) if DSflag else None

    print("Diversity Score: {}+-{}".format(*DS.mean_std()) if DSflag else "")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument("--train", action="store_true", default=False,
                        help='whether to use the training set (default=False)')
    parser.add_argument('--model_path', type=str,
                        help='which epoch to load')
    parser.add_argument('--sample_path', type=str, default='samples',
                        help='path to save generated images')
    parser.add_argument('-t', '--truncation_threshold', type=float, default=2.0,
                        help='the threshold of the truncation trick in sampling')
    parser.add_argument('-r', '--repeat', type=int, default=5,
                        help='the number of copies of a given layout')
    parser.add_argument('-D', '--DS', action="store_true", default=False,
                        help='whether to get DS scores')
    parser.add_argument('-N', '--NOSaving', action="store_true", default=False,
                        help='whether to save images')
    parser.add_argument('-C', '--cropped_size', type=int, default=0, help='')
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--image_id_savepath', type=str, default=None,
                        help='the txt to record the image order for YOLO scores')
    parser.add_argument('--gpu', type=str, default='0',
                        help='whick GPU to use')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
