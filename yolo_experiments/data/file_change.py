with open('train.txt','w',encoding='utf8')as fp:
    for i in range(0,3097):
        #j = str(i).zfill(6)
        #fp.writelines("../data/64_coco_layout2im/img"+j+".png\n")
        fp.writelines("../data/64_ours/sample_"+str(i)+".jpg\n")