import json
from .cocostuff_loader import CocoSceneGraphDataset
from .vg import VgSceneGraphDataset


def get_dataset(dataset='coco', img_size=0, left_right_flip=False, train=False):
    if train:
        if dataset == "coco":
            return_dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/train2017/',
                                            instances_json='./datasets/coco/annotations/instances_train2017.json',
                                            stuff_json='./datasets/coco/annotations/stuff_train2017.json',
                                            stuff_only=True, image_size=(img_size, img_size), left_right_flip=left_right_flip)
        elif dataset == 'vg':
            with open("./datasets/vg/vocab.json", "r") as read_file:
                vocab = json.load(read_file)
            return_dataset = VgSceneGraphDataset(vocab=vocab, h5_path='./datasets/vg/train.h5',
                                        image_dir='./datasets/vg/images/',
                                        image_size=(img_size, img_size), max_objects=10, left_right_flip=left_right_flip)
    else:
        if dataset == 'coco':
            return_dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/val2017/',
                                            instances_json='./datasets/coco/annotations/instances_val2017.json',
                                            stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                            stuff_only=True, image_size=(img_size, img_size), left_right_flip=left_right_flip)
        elif dataset == 'vg':
            with open("./datasets/vg/vocab.json", "r") as read_file:
                vocab = json.load(read_file)
            return_dataset = VgSceneGraphDataset(vocab=vocab,
                                        h5_path='./datasets/vg/val.h5',
                                        image_dir='./datasets/vg/images/',
                                        image_size=(img_size, img_size), 
                                        left_right_flip=left_right_flip, max_objects=10)
    return return_dataset