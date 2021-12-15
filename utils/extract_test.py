import os
import json
import pathlib
from PIL import Image
from tqdm import tqdm
import imageio
import numpy as np
import torch
from data.vg import VgSceneGraphDataset

# this is to extract testset images in the given sizes


def main(args):
    img_size = args.img_size
    save_path = f"datasets/{args.dataset}/test_{img_size}" 
    print(f"Saved in {save_path}")
    os.makedirs(save_path, exist_ok=True)

    if args.dataset == "coco":
        path = pathlib.Path(args.coco_test_path)
        files = list(path.glob('**/*.jpg')) + list(path.glob('**/*.png'))
        for idx, x in tqdm(enumerate(files)):
            x = Image.open(str(x)).convert("RGB")
            x = x.resize( (img_size, img_size) )
            x = np.array(x).astype(np.uint8)
            imageio.imwrite(f"{save_path}/sample_{idx}.png", x)
    elif args.dataset == "vg":
        with open("./datasets/vg/vocab.json", "r") as read_file:
                        vocab = json.load(read_file)
        return_dataset = VgSceneGraphDataset(vocab=vocab,
                                    h5_path='./datasets/vg/test.h5',
                                    image_dir='./datasets/vg/images/',
                                    image_size=(img_size, img_size), 
                                    left_right_flip=False, max_objects=30)
        dataloader = torch.utils.data.DataLoader(
                    return_dataset, batch_size=1,
                    drop_last=False, shuffle=False, num_workers=1)
        for idx, data in tqdm(enumerate(dataloader)):
            real_image, _, _ = data
            imageio.imwrite(f"{save_path}/sample_{idx}.png",
                        real_image[0].mul_(0.5).add_(0.5).mul_(255).to(torch.uint8).numpy().transpose(1, 2, 0))
    else:
        pass

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--coco_test_path", type=str, default="datasets/coco/test2017",
        help='Path to the extract images')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='image size')
    args = parser.parse_args()
    main(args)

# python extract_coco_test.py --dataset coco --img_size 64
# python utils/extract_coco_test.py --dataset coco --img_size 256
# python utils/extract_test.py --dataset vg --img_size 256