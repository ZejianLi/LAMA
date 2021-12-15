import argparse
import json
import os

import imageio
from PIL import Image, ImageDraw
from tqdm import tqdm

import sys, os
sys.path.append(os.getcwd()) 
from data.cocostuff_loader import *
from data.vg import *
from data import get_dataset


def get_dataloader(dataset = 'coco', img_size=0, train=False):
    dataset = get_dataset(dataset, img_size, train=train, left_right_flip=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1,
                    drop_last=False, shuffle=False, num_workers=0)
    return dataloader

def main(args):
    num_classes = 184 if 'coco' in args.dataset  else 179
    num_o = 8 if 'coco' in args.dataset else 31 
    dataloader = get_dataloader(args.dataset, args.img_size, args.train)

    train_val = "train" if args.train else "val"
    save_path = f"./datasets/{args.dataset}/{train_val}_{args.img_size}_cropped_{args.cropped_size}"
    print(f"Saved in {save_path}")
    for idx in range(num_classes):
        os.makedirs(os.path.join(save_path, str(idx)), exist_ok = True)

    for idx, data in tqdm(enumerate(dataloader)):
        real_image, label, bbox = data
        bbox[:, :, 2] = (bbox[:, :, 2] + bbox[:, :, 0]).mul(real_image.size(3))  
        bbox[:, :, 3] = (bbox[:, :, 3] + bbox[:, :, 1]).mul(real_image.size(2))
        bbox[:, :, 0] = bbox[:, :, 0].mul(real_image.size(3))
        bbox[:, :, 1] = bbox[:, :, 1].mul(real_image.size(2))
        real_image = real_image[0].mul_(0.5).add_(0.5).mul_(255).to(torch.uint8).numpy().transpose(1, 2, 0)
        realImage = Image.fromarray(real_image)
        for idx1, bl in enumerate(zip(bbox[0], label[0].tolist())):
            b, l = bl[0].ceil().tolist(), bl[1]
            if l>0:
                cropped = realImage.crop(b).resize( (args.cropped_size, args.cropped_size), resample=Image.BICUBIC )
                cropped.save(os.path.join(save_path, str(l), f"{idx}_{idx1}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='dataset to be cropped')
    parser.add_argument('--img_size', type=int, default=64,
                        help='resized images')
    parser.add_argument('--cropped_size', type=int, default=224,
                        help='resized cropped images')
    parser.add_argument('--train', action="store_true", default=False, 
                        help='using the training set')
    
    args = parser.parse_args()
    main(args)

# python utils/extract_cropped_objects.py --dataset coco --img_size 64 --cropped_size 224
# python utils/extract_cropped_objects.py --dataset coco --img_size 128 --cropped_size 224
# python utils/extract_cropped_objects.py --dataset coco --img_size 128 --cropped_size 32
# python utils/extract_cropped_objects.py --dataset coco --train --img_size 128 --cropped_size 32
# python utils/extract_cropped_objects.py --dataset vg   --img_size 64 --cropped_size 224
# python utils/extract_cropped_objects.py --dataset vg   --img_size 128 --cropped_size 224
# python utils/extract_cropped_objects.py --dataset coco --img_size 256 --cropped_size 224
# python utils/extract_cropped_objects.py --dataset vg --img_size 256 --cropped_size 224