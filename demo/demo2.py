import matplotlib.pyplot as plt
import os
import glob
import cv2
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np

import json
# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12
from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
config_file = "./configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
if not os.path.exists('results'):
    os.makedirs('results')

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)
infer_path = './datasets/jinnan2/jinnan2_round2_test_a_20190401'
# infer_path = './datasets/jinnan/jinnan2_round1_train_20190305/normal'
images = glob.glob('{}/*.jpg'.format(infer_path))


det_path = 'predictions/prediction'

for image in images:

    file_info = []
    basename = os.path.basename(image)
    
    #保存图像名
    img_name = basename[:-4]
    # print(img_name)
    save_path = 'results/' + basename.replace('jpg', 'png')

    image = cv2.imread(image)    

    predictions, confidence, boxes, labels, masks = coco_demo.run_on_opencv_image(image)
    print(basename)  # 文件名，如163.jpg
    # print(type(confidence))  # [c1, c2 ...]
    # print(len(boxes))  # [[xyxy], []...]
    # print(labels)
    # print((masks.shape))
    # cv2.imwrite(save_path, predictions)
    
    #图像高和宽
    height = len(image)
    width = len(image[0])
    
    #定义全零mask
    masks_1 =  np.zeros((height, width))
    masks_2 =  np.zeros((height, width))
    masks_3 =  np.zeros((height, width))
    masks_4 =  np.zeros((height, width))
    masks_5 =  np.zeros((height, width))
    
    
    # print(masks_1.shape)
    for i in range(0, len(labels)):
        if(labels[i] == 1):
            #求numpy元素的并集
            masks_1 = np.array((masks_1+masks[i][0]) >= 1).astype(int)
        elif(labels[i] == 2):
            masks_2 = np.array((masks_2+masks[i][0]) >= 1).astype(int)    
        elif(labels[i] == 3):
            masks_3 = np.array((masks_3+masks[i][0]) >= 1).astype(int) 
        elif(labels[i] == 4):
            masks_4 = np.array((masks_4+masks[i][0]) >= 1).astype(int) 
        elif(labels[i] == 5):
            masks_5 = np.array((masks_5+masks[i][0]) >= 1).astype(int) 
    
    #定义mask保存路径
    masks_1_pth =  det_path + '/' + img_name + '_1'
    masks_2_pth =  det_path + '/' + img_name + '_2'
    masks_3_pth =  det_path + '/' + img_name + '_3'
    masks_4_pth =  det_path + '/' + img_name + '_4'
    masks_5_pth =  det_path + '/' + img_name + '_5'
    
    #保存mask文件
    np.save(masks_1_pth, masks_1)
    np.save(masks_2_pth, masks_2)
    np.save(masks_3_pth, masks_3)
    np.save(masks_4_pth, masks_4)
    np.save(masks_5_pth, masks_5)
