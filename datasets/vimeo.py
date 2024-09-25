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

class vimeo_dataset():
    def __init__(self, type, args, dataset_path, dataset):
        self.train_type = type
        self.dataset = dataset
        self.dataset_path = dataset_path    

        if self.dataset == "vimeo_triplet":
            if self.train_type == 'train':
                self.dataset_list = [os.path.join(self.dataset_path, '01_' + self.dataset, 'sequences')]
            else:
                self.dataset_list = [os.path.join(self.dataset_path, '01_' + self.dataset, 'sequences')]
                

            train_fn = os.path.join(self.dataset_path, '01_' + self.dataset, 'tri_trainlist.txt')
            test_fn = os.path.join(self.dataset_path, '01_' + self.dataset, 'tri_testlist.txt')

        if self.train_type == 'train':
            with open(train_fn, 'r') as f:
                self.trainlist = f.read().splitlines()
        else:
            with open(test_fn, 'r') as f:
                self.testlist = f.read().splitlines()
        
    def __len__(self):
        if self.train_type == 'train':
            return len(self.trainlist)
        else:
            return len(self.testlist)
        
    def getimg(self, index):
        if self.train_type == 'train':        
            try:
                index = index % len(self.trainlist) if index >= len(self.trainlist) else index
            except:
                print('index: {}, dataset: {}, len(self.trainlist): {}\n'.format(index, self.dataset, len(self.trainlist)))
            img_path = self.trainlist[index]
            if self.dataset == "vimeo_triplet":
                selected_dataset = 0
            else:
                selected_dataset = 0    
            img_path = os.path.join(self.dataset_list[selected_dataset], img_path)
        
        else:
            try:
                index = index % len(self.testlist) if index >= len(self.testlist) else index
            except:
                print('index: {}, dataset: {}, len(self.trainlist): {}\n'.format(index, self.dataset, len(self.testlist)))
            
            img_path = self.testlist[index]
            img_path = os.path.join(self.dataset_list[0], img_path)        
            
        if self.dataset == 'vimeo_triplet':
            img_paths = [img_path + '/im1.png', img_path + '/im2.png', img_path + '/im3.png']
            
            img0 = cv2.imread(img_paths[0])
            gt = cv2.imread(img_paths[1])
            img1 = cv2.imread(img_paths[2])
                        
            return img0, gt, img1
            