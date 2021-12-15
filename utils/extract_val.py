import json, sys, os, argparse

import imageio
from tqdm import tqdm

sys.path.append(os.getcwd()) 
from data.cocostuff_loader import *
from data.vg import *
from data import get_dataset

def get_dataloader(dataset = 'coco', img_size=128):
    dataset = get_dataset(dataset, img_size, left_right_flip=False, train=False)
    dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=1,
                    drop_last=False, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    img_size = args.img_size
    save_path = f"datasets/{args.dataset}/val_{img_size}" 
    os.makedirs(save_path, exist_ok=True)

    dataloader = get_dataloader(args.dataset, img_size)

    for idx, data in tqdm(enumerate(dataloader)):
        real_image, _, _ = data
        imageio.imwrite(f"{save_path}/sample_{idx}.png",
                    real_image[0].mul_(0.5).add_(0.5).mul_(255).to(torch.uint8).numpy().transpose(1, 2, 0))
    print(f"Saved in {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='image size')
    args = parser.parse_args()
    main(args)

# python extract_val.py --dataset coco --img_size 128
# python utils/extract_val.py --dataset vg --img_size 64
# python utils/extract_val.py --dataset coco --img_size 256
# python utils/extract_val.py --dataset vg --img_size 256