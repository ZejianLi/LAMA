from PIL import Image
import os
import numpy as np


# file_list = os.listdir('val2017')
# for i in range (0,len(file_list)):
#     file_name = file_list[i].split('.')[0]
#     file_list[i] = (int(file_name))
# #print(file_list)
# temp = np.array(file_list)
# np.savetxt('coco_resize_id.txt',temp)    
# image_ID = np.loadtxt('coco_resize_id.txt')
# image_ID = image_ID.astype(int)
# 
#     for j in range(len(image_ID)):
#         fp.write('resize_val2017/'+str(image_ID[j]).zfill(12) + '.jpg\n')

#resize pic
image_ID = np.loadtxt('image_id.txt')
image_ID = image_ID.astype(int)
# file_list = os.listdir('../../experiments/for_yolo/with_reconstruction/coco128_repeat1_thres2.0/')
# print(len(image_ID))
for i in range (0,len(image_ID)):
    img = Image.open('../../SPADE/results/coco_pretrained/test_latest/images/synthesized_image/'+str(image_ID[i]).zfill(12)+'.png')
    # temp = file_list[i].split('_')
    print(i)
    # print(temp)
    # exit()
    #out1 = img.resize((64,64))
    out2 = img.resize((512,512))
    out2.save('SPADE/'+str(image_ID[i]).zfill(12)+'.jpg')

#save name to txt
image_ID = np.loadtxt('image_id.txt')
image_ID = image_ID.astype(int)
# print(len(file_list))
with open('train.txt','w') as fp: 
    for i in range(len(image_ID)):
        fp.write('../data/SPADE/'+str(image_ID[i]).zfill(12)+'.jpg\n')
# temp = np.array(file_list)
# np.savetxt('coco_resize_id.txt',temp)    