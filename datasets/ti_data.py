import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import torchvision.transforms.functional as F
import time

class ti_dataset():
    def __init__(self, train_type, args, dataset_path, threshold1, threshold2, siti_type=None):
        
        self.train_type = train_type
        self.dataset_path = dataset_path  #../split_data
        self.siti_type = siti_type
        self.threshold1 = threshold1
        self.threshold2 = threshold2
            
        if self.siti_type == 'easy':
            train_fn = os.path.join(self.dataset_path, 'train_text', 'vimeo_x1x2_TI_{}_{}'.format(threshold1, threshold2), 'easy.txt')
        elif self.siti_type == 'medium':
            train_fn = os.path.join(self.dataset_path, 'train_text', 'vimeo_x1x2_TI_{}_{}'.format(threshold1, threshold2), 'medium.txt')
        elif self.siti_type == 'hard':
            train_fn = os.path.join(self.dataset_path, 'train_text', 'vimeo_x1x2_TI_{}_{}'.format(threshold1, threshold2), 'hard.txt')
        else:   # all triplet 
            train_fn = os.path.join(self.dataset_path, 'train_text', 'vimeo_x1x2_TI_{}_{}'.format(threshold1, threshold2), 'train_all_list.txt')  
            print('train all patch')

    
        # read dataset dir list
        with open(train_fn, 'r') as f:
            self.train_list = f.read().splitlines()
        
        cnt = int(len(self.train_list) * 0.95)

        if self.train_type == 'train':
            self.trainlist = self.train_list[:cnt]
        else:
            self.trainlist = self.train_list[cnt:]

    def __len__(self):
        return len(self.trainlist)

        
    def getimg(self, index):   
        try:
            index = index % len(self.trainlist) if index >= len(self.trainlist) else index
        except:
            print('index: {}, dataset: {}, len(self.trainlist): {}\n'.format(index, self.dataset_path, len(self.trainlist)))
            
        img_path = self.trainlist[index]
        img_path = os.path.join(self.dataset_path, img_path) 
        
        if img_path.startswith(os.path.join(self.dataset_path, "05")):
            # print('find {}'.format(img_path))
            img_paths = [img_path + '/im1.jpg', img_path + '/im2.jpg', img_path + '/im3.jpg']

        else:
            img_paths = [img_path + '/im1.png', img_path + '/im2.png', img_path + '/im3.png']
            
        # load images
        img0 = cv2.imread(img_paths[0])
        gt = cv2.imread(img_paths[1])
        img1 = cv2.imread(img_paths[2])
        
            
        return img0, gt, img1
