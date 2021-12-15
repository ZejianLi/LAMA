from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np

annType = ['segm','bbox','keypoints']
annType = annType[1]
cocoGt = COCO("instances_val2017.json")
cocoDt=cocoGt.loadRes("Spade_result_final.json")
image_ID = np.loadtxt('image_id.txt')
image_ID = image_ID.astype(int)
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = image_ID
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
