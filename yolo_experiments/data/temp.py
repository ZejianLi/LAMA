import json
import numpy as np
from PIL import Image

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

            

with open('Spade_result.json','r',encoding='utf8')as fp:
    yolo_data = json.load(fp)
#print(json_data)

#image_ID = np.loadtxt('image_id.txt')
image_ID = np.loadtxt('image_id.txt')
image_ID = image_ID.astype(int)
#print(image_ID[0])

coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

coco_id = {value:key for key, value in coco_id_name_map.items()}
#print(coco_id)
#result = [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},{"image_id":73,"category_id":11,"bbox":[61,22.75,504,609.67],"score":0.318},{"image_id":73,"category_id":4,"bbox":[12.66,3.32,268.6,271.91],"score":0.726}]
result = []
elements = {}
#print(len(yolo_data[0]['objects']))
for i in range(0,len(yolo_data)):
    id = image_ID[i]
    id = str(id).zfill(12)
    im = Image.open("../../LostGANs/datasets/coco/val2017/" + id + ".jpg")
    #im = Image.open()
    (orgin_width,orgin_height) = im.size
    for j in range(0,len(yolo_data[i]['objects'])):
        elements["image_id"] = image_ID[i]
        if yolo_data[i]['objects'][j]['name'] == 'diningtable':
            elements["category_id"] = coco_id['dining table']
        elif yolo_data[i]['objects'][j]['name'] == 'tvmonitor':
            elements["category_id"] = coco_id['tv']
        elif yolo_data[i]['objects'][j]['name'] == 'pottedplant':
            elements["category_id"] = coco_id['potted plant']
        elif yolo_data[i]['objects'][j]['name'] == 'aeroplane':
            elements["category_id"] = coco_id['airplane']
        elif yolo_data[i]['objects'][j]['name'] == 'motorbike':
            elements["category_id"] = coco_id['motorcycle']
        elif yolo_data[i]['objects'][j]['name'] == 'sofa':
            elements["category_id"] = coco_id['couch']
        else:
            elements["category_id"] = coco_id[yolo_data[i]['objects'][j]['name']]
        
        temp_x = yolo_data[i]['objects'][j]['relative_coordinates']['center_x'] - 0.5 * yolo_data[i]['objects'][j]['relative_coordinates']['width']
        temp_y = yolo_data[i]['objects'][j]['relative_coordinates']['center_y'] - 0.5 * yolo_data[i]['objects'][j]['relative_coordinates']['height']
        temp_w = yolo_data[i]['objects'][j]['relative_coordinates']['width']
        temp_h = yolo_data[i]['objects'][j]['relative_coordinates']['height']
        elements["bbox"] = [temp_x * orgin_width,
                            temp_y * orgin_height,
                            temp_w * orgin_width,
                            temp_h * orgin_height]        
        elements["score"] = round(yolo_data[i]['objects'][j]['confidence'],3)
        #print(elements)
        result.append(elements.copy())

with open('Spade_result_final.json','w') as f:
    json.dump(result,f,cls=MyEncoder) 
