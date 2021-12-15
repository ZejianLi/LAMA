from utils.bbox import coco_cats

def main(args):
    # parameters
    img_size = args.img_size
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
    save_path = args.sample_path
    print("Saved in {}".format(save_path))

    bbox, label = read_bbox(args.bbox_path, img_size=args.img_size)

    z_obj = torch.from_numpy(truncted_random(num_o=bbox.size(1), thres=thres)).float().cuda()
    z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()
    with torch.no_grad():
        fake_image = netG.forward(z_obj, bbox.float().cuda(), z_im, label.squeeze(dim=-1).cuda())


def truncted_random(num_o=8, thres=1.0, dim=128):
    # according to BigGAN to resample latent codes when falling outside the threshold
    z = np.random.randn(1, num_o, dim)
    truncation_flag = (abs(z)>thres).astype(np.float32)
    z = truncation_flag * np.random.randn(1, num_o, dim) + (1.-truncation_flag) * z
    return z

def read_bbox(path, img_size=256):
    """
    Read the bbox configuration from {path} and return {bbox} and {label} specifying bbox
    """
    bbox = np.stack([np.ndarray([x,y,w,h])/img_size for x,y,w,h in l])
    label = torch.tensor( coco_cats[s] for s in tmp)
    return bbox, label

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
    parser.add_argument('-r', '--repeat', type=int, default=1,
                        help='the number of copies of a given layout')
    parser.add_argument('-D', '--DS', action="store_true", default=False,
                        help='whether to get DS scores')
    parser.add_argument('-N', '--NOSaving', action="store_true", default=False,
                        help='whether to save images')
    parser.add_argument('-C', '--cropped_size', type=int, default=0, help='')
    parser.add_argument('--gpu', type=str, default='0',
                        help='whick GPU to use')    
    parser.add_argument('--img_size', type=int, default=128, help='image size')
    args = parser.parse_args()
