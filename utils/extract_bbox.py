import json, sys, os, argparse

import imageio
from tqdm import tqdm

sys.path.append(os.getcwd()) 
from data.cocostuff_loader import *
from data.vg import *
from data import get_dataset
from utils import draw_bounding_box

def get_dataloader(dataset = 'coco', img_size=128):
    dataset = get_dataset(dataset, img_size, left_right_flip=False, train=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1,
                    drop_last=False, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    img_size = args.img_size
    save_path = f"datasets/{args.dataset}/bbox_{img_size}" 
    os.makedirs(save_path, exist_ok=True)
    dataloader = get_dataloader(args.dataset, img_size)

    for idx, data in tqdm(enumerate(dataloader)):
        real_image, label, bbox = data
        img = draw_bounding_box(torch.ones_like(real_image[0]), 
                bbox[0][:args.max_obj_num], 
                label.flatten()[:args.max_obj_num], 
                img_size=img_size, 
                outImage=True)
        img.save(os.path.join(save_path, f"sample_{idx}_bbox.png"))

    print(f"Saved in {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='image size')
    parser.add_argument('--max_obj_num', type=int, default=20,
                        help='image size')
    args = parser.parse_args()
    main(args)

# python utils/extract_bbox.py --dataset coco --img_size 512