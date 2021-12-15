import argparse, os

# this is to compute SceneFID in OCGAN
def main(args):
    cropped_size = 224
    data_path = f'datasets/{args.dataset}/val_{args.img_size}_cropped_{cropped_size}'
    assert os.path.exists(data_path), "Cropped objects may not be exported."
    print(f"Use real cropped samples in {data_path}")
    sample_path = os.path.join(args.intermediate_path, f'cropped_{cropped_size}')

    # sample from test.py or use existing samples in {sample_path}
    if args.model_path != '' and not os.path.exists(sample_path):
        generate_cmd = f"~/anaconda3/envs/zl6/bin/python test.py --dataset {args.dataset} -t 2.0 -r 5 --model_path {args.model_path} --sample_path {sample_path} --img_size {args.img_size} --gpu {args.gpu} -N --cropped_size {cropped_size}"
        print(generate_cmd)
        os.popen(generate_cmd).readlines()
    else:
        print(f"Use generated cropped samples in {sample_path}")

    # calculate FID
    fid_cmd = f"~/anaconda3/envs/zl_tf/bin/python scores/FID.py {data_path} {sample_path} --gpu {args.gpu} --lowprofile"
    print(fid_cmd)
    tmp = os.popen(fid_cmd).readlines()
    print(tmp)
    value = tmp[-1].split(' ')[-1]
    print(float(value[:9]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
            help='model_path')
    parser.add_argument('--intermediate_path', type=str, default='', 
            help='intermediate_path')
    parser.add_argument('--dataset', type=str, default='coco',
            help='training dataset')
    parser.add_argument('--img_size', type=int, default=128, 
            help='image size')
    parser.add_argument("--gpu", default="", type=str,
            help='GPU to use')
    args = parser.parse_args()

    main(args)
