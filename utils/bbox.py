from random import randint

import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor

coco_cats = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush', 92: 'banner', 93: 'blanket', 94: 'branch', 95: 'bridge', 96: 'building-other', 97: 'bush', 98: 'cabinet', 99: 'cage', 100: 'cardboard', 101: 'carpet', 102: 'ceiling-other', 103: 'ceiling-tile', 104: 'cloth', 105: 'clothes', 106: 'clouds', 107: 'counter', 108: 'cupboard', 109: 'curtain', 110: 'desk-stuff', 111: 'dirt', 112: 'door-stuff', 113: 'fence', 114: 'floor-marble', 115: 'floor-other', 116: 'floor-stone', 117: 'floor-tile', 118: 'floor-wood', 119: 'flower', 120: 'fog', 121: 'food-other', 122: 'fruit', 123: 'furniture-other', 124: 'grass', 125: 'gravel', 126: 'ground-other', 127: 'hill', 128: 'house', 129: 'leaves', 130: 'light', 131: 'mat', 132: 'metal', 133: 'mirror-stuff', 134: 'moss', 135: 'mountain', 136: 'mud', 137: 'napkin', 138: 'net', 139: 'paper', 140: 'pavement', 141: 'pillow', 142: 'plant-other', 143: 'plastic', 144: 'platform', 145: 'playingfield', 146: 'railing', 147: 'railroad', 148: 'river', 149: 'road', 150: 'rock', 151: 'roof', 152: 'rug', 153: 'salad', 154: 'sand', 155: 'sea', 156: 'shelf', 157: 'sky-other', 158: 'skyscraper', 159: 'snow', 160: 'solid-other', 161: 'stairs', 162: 'stone', 163: 'straw', 164: 'structural-other', 165: 'table', 166: 'tent', 167: 'textile-other', 168: 'towel', 169: 'tree', 170: 'vegetable', 171: 'wall-brick', 172: 'wall-concrete', 173: 'wall-other', 174: 'wall-panel', 175: 'wall-stone', 176: 'wall-tile', 177: 'wall-wood', 178: 'water-other', 179: 'waterdrops', 180: 'window-blind', 181: 'window-other', 182: 'wood', 183: 'other'}

def mat_inter(box1, box2):
    """
    whether two bbox is overlapped
    """
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
 
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def draw_bounding_box(image, bbox, label, dataset='coco', img_size=(128, 128), transform_reverse=True, outImage=False):
    """
    draw bounding boxes over {image} according to {bbox}, {label} and the categories in {dataset}.
    the inputs should in one sample, and the batch dimension is eliminated
    - image : a 3*w*h torch tensor if {transform_reverse} or a numpy array elsewise
    - bbox  : num_o * 4
    - label : num_o * 1
    """
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    num_classes = 184 if 'coco' in dataset  else 179
    colormap = (np.array(sns.color_palette("deep", n_colors=num_classes, desat=.6))*255).astype(np.int16)

    if transform_reverse:
        image = image.cpu().detach().mul(0.5).add(0.5).mul(255).numpy().astype(np.uint8).transpose(1, 2, 0)
    I = Image.fromarray(image)
    # get a drawing context
    d = ImageDraw.Draw(I)
    # the list of text positions
    text_position_list = []
    text_list = []
    color_list = []
    

    ft = ImageFont.truetype("Helvetica.ttc", 32)

    # for each bbox
    for bb, l in zip(bbox, label):
        if l>0:
            # position translation
            bbox_x, bbox_w = bb[0]*img_size[0], bb[2]*img_size[0]
            bbox_y, bbox_h = bb[1]*img_size[1], bb[3]*img_size[1]
            x = [bbox_x, bbox_y, bbox_x+bbox_w, bbox_y+bbox_h]
            color = tuple(colormap[l.item()])
            text = coco_cats[l.item()]
            # draw bbox rectangle
            d.rectangle(x, outline=color, width=8)
            # the text width and height
            text_width, text_height = d.textsize(text, font=ft)
            text_position = [bbox_x, bbox_y, bbox_x+text_width, bbox_y+text_height]
            # adjust the text position
            for _ in range(3):
                # if overlap
                if any(mat_inter(text_position, lll) for lll in text_position_list):
                    jump = text_height + randint(1, 8)
                    text_position[1] += jump
                    text_position[3] += jump
                else:
                    break
            # record the position
            text_position_list.append(text_position)
            text_list.append(text)
            color_list.append(color)

        for text_position, text, color in zip(text_position_list, text_list, color_list):
            # draw text
            d.rectangle(text_position, outline=color, fill=color)
            d.text(text_position[:2], text, (255,)*3, font=ft)

    return I if outImage else ToTensor()(I)


# # # example
# from utils import draw_bounding_box
# from test import get_dataloader
# size = 512
# iii = iter(get_dataloader(dataset = 'coco', img_size=size))
# image, label, bbox = next(iii)
# draw_bounding_box(image[0], bbox[0], label[0], img_size=size, outImage=True).save("ttt.jpg")
