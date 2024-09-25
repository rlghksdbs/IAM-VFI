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
from torchvision.transforms import CenterCrop

cv2.setNumThreads(8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader(Dataset):
    def __init__(self, type, args, arbitrary=False):
        self.train_type = type
        self.arbitrary = arbitrary
        self.dataset = args.datasets
        self.dataset_path = args.datasets_path
        self.batch_size = args.batch_size
        self.mix_dataset = args.mix_dataset
        self.vimeo_class_type = args.vimeo_class_type    # for ClassVFI

        #TI dataset
        self.split_data_path = args.split_data_path
        self.ti_type = args.ti_type
        self.threshold1 = args.ti_threshold1
        self.threshold2 = args.ti_threshold2

        # argumentation parameters        
        self.patch_size_x = args.patch_size_x
        self.patch_size_y = args.patch_size_y
        self.rotate90 = args.rotate90
        self.rotate = args.rotate
        self.contrast = args.contrast
        self.brightness = args.brightness
        self.gamma = args.gamma
        self.hue = args.hue
        self.saturation = args.saturation
        self.zoom = args.zoom
        self.aug_prob = args.aug_prob
        self.mosaic_boarder = [self.patch_size_x // 2, self.patch_size_y]
        self.cut_mix_prob = args.cut_mix

        self.dataset_class_list = []
        
        if self.train_type == 'train':
            if 'vimeo_triplet' in self.dataset:
                from datasets.vimeo import vimeo_dataset
                self.dataset_class_list.append(vimeo_dataset(self.train_type, args, self.dataset_path, self.dataset[self.dataset.index("vimeo_triplet")], self.vimeo_class_type, self.arbitrary, mix_dataset=self.mix_dataset))
            if 'ti_dataset' in self.dataset:
                from datasets.ti_data import ti_dataset
                self.dataset_class_list.append(ti_dataset(self.train_type, args, self.split_data_path, self.threshold1, self.threshold2, self.ti_type))
        else:   # validation
            if 'ti_dataset' in self.dataset:
                from datasets.ti_data import ti_dataset
                self.dataset_class_list.append(ti_dataset(self.train_type, args, self.split_data_path, self.threshold1, self.threshold2, self.ti_type))
            else:
                from datasets.vimeo import vimeo_dataset
                self.dataset_class_list.append(vimeo_dataset(self.train_type, args, self.dataset_path, "vimeo_triplet", arbitrary=self.arbitrary, mix_dataset=self.mix_dataset))
        
    def __len__(self):        
        sum=0
        for i in range(len(self.dataset_class_list)):
            sum += len(self.dataset_class_list[i])
        return sum

    def __getitem__(self, index):            
        selected_dataset = random.randint(0, len(self.dataset_class_list)-1)
        
        if self.arbitrary == True:            
            img0, gt, img1, timestep = self.dataset_class_list[selected_dataset].getimg(index)
        else:
            img0, gt, img1 = self.dataset_class_list[selected_dataset].getimg(index)
        
        if self.train_type == 'train':
            if img0.shape[0] < self.patch_size_y:
                img0, gt, img1 = self.aug_pad_ver(img0, gt, img1, self.patch_size_y, self.patch_size_x)
            elif img0.shape[1] < self.patch_size_x:
                img0, gt, img1 = self.aug_pad_hor(img0, gt, img1, self.patch_size_y, self.patch_size_x)
            else:
                img0, gt, img1 = self.aug(img0, gt, img1, self.patch_size_y, self.patch_size_x)

            if random.uniform(0, 1) < self.cut_mix_prob:
                img0, img1, gt = self.cut_mix(img0, img1, gt)

            if random.uniform(0, 1) < 0.5:  # RGB to BGR
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:  # Vertical
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:  # Horizon
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                

            if self.rotate90 and random.uniform(0, 1) < 0.5:  # 90 Rotate
                img0 = np.rot90(img0)
                img1 = np.rot90(img1)
                gt = np.rot90(gt)

            img0 = Image.fromarray(img0)
            img1 = Image.fromarray(img1)
            gt = Image.fromarray(gt)

            if self.rotate and random.uniform(0, 1) < 0.5:  # 0 ~ 89 Rotate
                angle = np.random.randint(0, 90)
                img0 = img0.rotate(angle)
                img1 = img1.rotate(angle)
                gt = gt.rotate(angle)

            if self.brightness > 0. and random.uniform(0, 1) < self.aug_prob:
                brightness_str = random.uniform(1 - self.brightness, 1 + self.brightness)
                img0 = F.adjust_brightness(img0, brightness_factor=1-brightness_str)
                img1 = F.adjust_brightness(img1, brightness_factor=1+brightness_str)
                # gt = F.adjust_brightness(gt, brightness_factor=brightness_str)

            if self.contrast > 0. and random.uniform(0, 1) < self.aug_prob:
                contrast_str = random.uniform(1 - self.contrast, 1 + self.contrast)
                img0 = F.adjust_contrast(img0, contrast_factor=contrast_str)
                img1 = F.adjust_contrast(img1, contrast_factor=contrast_str)
                gt = F.adjust_contrast(gt, contrast_factor=contrast_str)

            if self.saturation > 0. and random.uniform(0, 1) < self.aug_prob:
                saturation_str = random.uniform(1 - self.saturation, 1 + self.contrast)
                img0 = F.adjust_saturation(img0, saturation_factor=saturation_str)
                img1 = F.adjust_saturation(img1, saturation_factor=saturation_str)
                gt = F.adjust_saturation(gt, saturation_factor=saturation_str)

            if self.hue > 0. and random.uniform(0, 1) < self.aug_prob:
                hue_str = random.uniform(-self.hue, self.hue)
                img0 = F.adjust_hue(img0, hue_factor=hue_str)
                img1 = F.adjust_hue(img1, hue_factor=hue_str)
                gt = F.adjust_hue(gt, hue_factor=hue_str)

            img0 = np.asarray(img0)
            img1 = np.asarray(img1)
            gt = np.asarray(gt)

            if random.uniform(0, 1) < 0.5:  # Reverse
                tmp = img1
                img1 = img0
                img0 = tmp
                if self.arbitrary == True:
                    timestep = 1- timestep
                    

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

        if self.train_type != 'train':
            c, h, w = img0.shape
            if 256 != w or 256 != h:
                img0 = CenterCrop(size=(256, 256))(img0)
                img1 = CenterCrop(size=(256, 256))(img1)
                gt = CenterCrop(size=(256, 256))(gt)
                
        if self.arbitrary == True:
            return torch.cat((img0, img1, gt), 0), timestep
        else:
            return torch.cat((img0, img1, gt), 0)
        
    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def aug_pad_ver(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        padding_size = (h - ih) // 2
        ver_pad = ((padding_size, padding_size), (0, 0), (0, 0))
        #x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[:, y:y+w, :]
        img1 = img1[:, y:y+w, :]
        gt = gt[:, y:y+w, :]
        img0 = np.pad(img0, ver_pad, 'constant', constant_values=0)
        img1 = np.pad(img1, ver_pad, 'constant', constant_values=0)
        gt = np.pad(gt, ver_pad, 'constant', constant_values=0)
        return img0, gt, img1


    def mosaic(self, img0, img1, gt, index):
        pass

    def normalize(self, img0, img1, gt):
        imgs0 = np.stack((img0[:, :, 0], img1[:, :, 0], gt[:, :, 0]), 2)
        imgs1 = np.stack((img0[:, :, 1], img1[:, :, 1], gt[:, :, 1]), 2)
        imgs2 = np.stack((img0[:, :, 2], img1[:, :, 2], gt[:, :, 2]), 2)

        mean0, std0 = np.mean(imgs0), np.std(imgs0)
        mean1, std1 = np.mean(imgs1), np.std(imgs1)
        mean2, std2 = np.mean(imgs2), np.std(imgs2)

        mean = np.array([mean0, mean1, mean2])
        std = np.array([std0, std1, std2])
        #print(mean, std)

        img0 = (img0 - mean) /(std + 0.001)
        img1 = (img1 - mean) / (std + 0.001)
        gt = (gt - mean) / (std + 0.001)

        return img0.astype(np.float32), img1.astype(np.float32), gt.astype(np.float32)

    def rand_bbox(self, W, H, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2 - bbx1, bby2 - bby1

    def cut_mix(self, img0, img1, gt):

        lam = random.uniform(0.25, 0.75)
        index_add = random.randint(0, self.__len__() - 1)
        # img0_add, gt_add, img1_add = self.getimg(index_add)
        selected_dataset = random.randint(0, len(self.dataset_class_list)-1)
        img0_add, gt_add, img1_add = self.dataset_class_list[selected_dataset].getimg(index_add)

        if img0_add.shape[0] < self.patch_size_y:
            img0_add, gt_add, img1_add = self.aug_pad_ver(img0_add, gt_add, img1_add, self.patch_size_y, self.patch_size_x)
        elif img0_add.shape[1] < self.patch_size_x:
            img0_add, gt_add, img1_add = self.aug_pad_hor(img0_add, gt_add, img1_add, self.patch_size_y, self.patch_size_x)
        else:
            img0_add, gt_add, img1_add = self.aug(img0_add, gt_add, img1_add, self.patch_size_y, self.patch_size_x)

        y_size, x_size, _ = img0_add.shape

        if random.uniform(0, 1) < 0.5:  # horizontal
            cut_x = random.randint(int(self.patch_size_x * 0.25), int(self.patch_size_x * 0.75))
            x_len = self.patch_size_x - cut_x
            y_len = self.patch_size_y

            x_begin = random.randint(0, x_size - x_len)
            y_begin = random.randint(0, y_size - y_len)

            img0[:, cut_x:cut_x + x_len, :] = img0_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            img1[:, cut_x:cut_x + x_len, :] = img1_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            gt[:, cut_x:cut_x + x_len, :]     = gt_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]

        else:  # vertical
            cut_y = random.randint(int(self.patch_size_y * 0.33), int(self.patch_size_y * 0.67))
            y_len = self.patch_size_y - cut_y
            x_len = self.patch_size_x

            x_begin = random.randint(0, x_size - x_len)
            y_begin = random.randint(0, y_size - y_len)

            img0[cut_y:cut_y + y_len, :, :] = img0_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            img1[cut_y:cut_y + y_len, :, :] = img1_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            gt[cut_y:cut_y + y_len, :, :]     = gt_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]

        return img0, img1, gt

    def load_mosaic(self, index):
        # loads images in a mosaic

        sx, sy = self.patch_size_x, self.patch_size_y
        xc = int(random.uniform(-self.mosaic_border[0], 2 * sx + x))
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in
                             range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            # img0, gt, img1 = self.getimg(index)
            selected_dataset = random.randint(0, len(self.dataset_class_list)-1)
            img0, gt, img1 = self.dataset_class_list[selected_dataset].getimg(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        return img4, labels4

class VimeoDataset(Dataset):
    def __init__(self, dataset_name, args, arbitrary=False, vimeo_septuplet_distance=[1]):
        self.dataset_name = dataset_name
        self.batch_size = args.batch_size
        self.patch_size_x = args.patch_size_x
        self.patch_size_y = args.patch_size_y
        self.rotate90 = args.rotate90
        self.rotate = args.rotate
        self.contrast = args.contrast
        self.brightness = args.brightness
        self.gamma = args.gamma
        self.hue = args.hue
        self.saturation = args.saturation
        self.zoom = args.zoom
        self.aug_prob = args.aug_prob
        self.mosaic_boarder = [self.patch_size_x // 2, self.patch_size_y]
        self.cut_mix_prob = args.cut_mix
        self.mix_dataset = args.mix_dataset.split(',') if args.mix_dataset is not None else []
        self.arbitrary = arbitrary
        self.vimeo90k_dir = args.datasets        
        self.vimeo_septuplet_distance = vimeo_septuplet_distance # septuplet 일 경우 im(0, 2)와 im1과의 오프셋 거리
        

        self.data_root = os.path.join('../data', self.vimeo90k_dir)
        self.data_root_x2 = '../data/vimeo_triplet_x2'
        self.data_root_x3 = '../data/vimeo_triplet_x3'
        self.data_root_x4 = '../data/vimeo_triplet_x4'
        self.val_data_root = os.path.join('../data', self.vimeo90k_dir)
        self.data_root_list = [self.data_root]
        for i in self.mix_dataset:
            if i == 'x2':
                self.data_root_list.append(self.data_root_x2)
            if i == 'x3':
                self.data_root_list.append(self.data_root_x3)
            if i == 'x4':
                self.data_root_list.append(self.data_root_x4)

        self.image_root = []
        for i in self.data_root_list:
            self.image_root.append(os.path.join(i, 'sequences'))
        #self.image_root = os.path.join(self.data_root, 'sequences')
        self.val_image_root = os.path.join(self.val_data_root, 'sequences')
        if self.vimeo90k_dir == 'vimeo_septuplet':
            train_fn = os.path.join(self.data_root, 'sep_trainlist.txt')
            test_fn = os.path.join(self.val_data_root, 'sep_testlist.txt')    
        else:
            train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
            test_fn = os.path.join(self.val_data_root, 'tri_testlist.txt')

        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        #print(test_fn)
        #print(self.testlist)
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        # cnt = int(len(self.trainlist) * 0.95)
        # self.dataset_name 의 종류: 'train', 'test', 'validation'
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist # [:cnt]   # 학습시 95%만 training set 으로 사용
        else:
            self.meta_data = self.testlist

    def aug(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1

    def aug_pad_ver(self, img0, gt, img1, h, w):
        ih, iw, _ = img0.shape
        padding_size = (h - ih) // 2
        ver_pad = ((padding_size, padding_size), (0, 0), (0, 0))
        #x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[:, y:y+w, :]
        img1 = img1[:, y:y+w, :]
        gt = gt[:, y:y+w, :]
        img0 = np.pad(img0, ver_pad, 'constant', constant_values=0)
        img1 = np.pad(img1, ver_pad, 'constant', constant_values=0)
        gt = np.pad(gt, ver_pad, 'constant', constant_values=0)
        return img0, gt, img1

    def getimg(self, index):
        imgpath = self.meta_data[index]
        if self.dataset_name == 'train':
            selected_dataset = random.randint(0, len(self.image_root) - 1)      # self.image_root(list) 에 여러 path 가 있는 경우(x1, x2, x3, x4) random 으로 폴도 목록 택 1
            imgpath = os.path.join(self.image_root[selected_dataset], imgpath)
        else:
            imgpath = os.path.join(self.val_image_root, imgpath)
    
        if self.vimeo90k_dir == 'vimeo_septuplet':
            imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png', imgpath + '/im4.png', imgpath + '/im5.png', imgpath + '/im6.png', imgpath + '/im7.png']
            if self.arbitrary == True:
                # RIFEm with Vimeo-septuplet
                ind = [0, 1, 2, 3, 4, 5, 6]
                random.shuffle(ind)
                ind = ind[:3]
                ind.sort()

                # Load images
                img0 = cv2.imread(imgpaths[ind[0]])
                gt = cv2.imread(imgpaths[ind[1]])
                img1 = cv2.imread(imgpaths[ind[2]])
                timestep = (ind[1] - ind[0]) * 1.0 / (ind[2] - ind[0] + 1e-6)

                # Get image shape for arbitrary img size
                self.h, self.w, _ = img0.shape
                return img0, gt, img1, timestep
            else:
                selected_distance = random.randint(0, len(self.vimeo_septuplet_distance) - 1)                
                if self.dataset_name == 'train':                    
                    distance = int(self.vimeo_septuplet_distance[selected_distance])                
                    if distance == 1:
                        start_frame_idx = random.randint(0, 4)
                    elif distance == 2:
                        start_frame_idx = random.randint(0, 2)
                    elif distance == 3:
                        start_frame_idx = 0 # frame idx 0, 3, 6 한가지 밖에 없음
                        
                    img0 = cv2.imread(imgpaths[start_frame_idx])
                    gt = cv2.imread(imgpaths[start_frame_idx + distance])
                    img1 = cv2.imread(imgpaths[start_frame_idx + distance * 2])
                
                else:
                    img0 = cv2.imread(imgpaths[2])
                    gt = cv2.imread(imgpaths[3])
                    img1 = cv2.imread(imgpaths[4])

                self.h, self.w = img0.shape[0], img0.shape[1]
                return img0, gt, img1
        else:
            # RIFE with Vimeo-trituplet
            imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

            # Load images
            img0 = cv2.imread(imgpaths[0])
            gt = cv2.imread(imgpaths[1])
            img1 = cv2.imread(imgpaths[2])

            # Get image shape for arbitrary img size
            self.h, self.w = img0.shape[0], img0.shape[1]
            
            if self.arbitrary == True:
                return img0, gt, img1, 0.5
            else:
                return img0, gt, img1

    def mosaic(self, img0, img1, gt, index):
        pass

    def normalize(self, img0, img1, gt):
        imgs0 = np.stack((img0[:, :, 0], img1[:, :, 0], gt[:, :, 0]), 2)
        imgs1 = np.stack((img0[:, :, 1], img1[:, :, 1], gt[:, :, 1]), 2)
        imgs2 = np.stack((img0[:, :, 2], img1[:, :, 2], gt[:, :, 2]), 2)

        mean0, std0 = np.mean(imgs0), np.std(imgs0)
        mean1, std1 = np.mean(imgs1), np.std(imgs1)
        mean2, std2 = np.mean(imgs2), np.std(imgs2)

        mean = np.array([mean0, mean1, mean2])
        std = np.array([std0, std1, std2])
        #print(mean, std)

        img0 = (img0 - mean) /(std + 0.001)
        img1 = (img1 - mean) / (std + 0.001)
        gt = (gt - mean) / (std + 0.001)

        return img0.astype(np.float32), img1.astype(np.float32), gt.astype(np.float32)

    def rand_bbox(self, W, H, lam):
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2 - bbx1, bby2 - bby1

    def cut_mix(self, img0, img1, gt):

        lam = random.uniform(0.25, 0.75)
        index_add = random.randint(0, self.__len__() - 1)
        img0_add, gt_add, img1_add = self.getimg(index_add)

        if img0_add.shape[0] < self.patch_size_y:
            img0_add, gt_add, img1_add = self.aug_pad_ver(img0_add, gt_add, img1_add, self.patch_size_y, self.patch_size_x)
        elif img0_add.shape[1] < self.patch_size_x:
            img0_add, gt_add, img1_add = self.aug_pad_hor(img0_add, gt_add, img1_add, self.patch_size_y, self.patch_size_x)
        else:
            img0_add, gt_add, img1_add = self.aug(img0_add, gt_add, img1_add, self.patch_size_y, self.patch_size_x)

        y_size, x_size, _ = img0_add.shape

        if random.uniform(0, 1) < 0.5:  # horizontal
            cut_x = random.randint(int(self.patch_size_x * 0.25), int(self.patch_size_x * 0.75))
            x_len = self.patch_size_x - cut_x
            y_len = self.patch_size_y

            x_begin = random.randint(0, x_size - x_len)
            y_begin = random.randint(0, y_size - y_len)

            img0[:, cut_x:cut_x + x_len, :] = img0_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            img1[:, cut_x:cut_x + x_len, :] = img1_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            gt[:, cut_x:cut_x + x_len, :]     = gt_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]

        else:  # vertical
            cut_y = random.randint(int(self.patch_size_y * 0.33), int(self.patch_size_y * 0.67))
            y_len = self.patch_size_y - cut_y
            x_len = self.patch_size_x

            x_begin = random.randint(0, x_size - x_len)
            y_begin = random.randint(0, y_size - y_len)

            img0[cut_y:cut_y + y_len, :, :] = img0_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            img1[cut_y:cut_y + y_len, :, :] = img1_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]
            gt[cut_y:cut_y + y_len, :, :]     = gt_add[y_begin:y_begin + y_len, x_begin:x_begin + x_len, :]

        return img0, img1, gt

    def load_mosaic(self, index):
        # loads images in a mosaic

        sx, sy = self.patch_size_x, self.patch_size_y
        xc = int(random.uniform(-self.mosaic_border[0], 2 * sx + x))
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
        indices = [index] + [self.indices[random.randint(0, self.n - 1)] for _ in
                             range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img0, gt, img1 = self.getimg(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        return img4, labels4

    def __getitem__(self, index):
        
        if self.arbitrary == True:
            img0, gt, img1, timestep = self.getimg(index)
        else:
            img0, gt, img1 = self.getimg(index)
            
        if self.dataset_name == 'train':
            if img0.shape[0] < self.patch_size_y:
                img0, gt, img1 = self.aug_pad_ver(img0, gt, img1, self.patch_size_y, self.patch_size_x)
            elif img0.shape[1] < self.patch_size_x:
                img0, gt, img1 = self.aug_pad_hor(img0, gt, img1, self.patch_size_y, self.patch_size_x)
            else:
                img0, gt, img1 = self.aug(img0, gt, img1, self.patch_size_y, self.patch_size_x)

            if random.uniform(0, 1) < self.cut_mix_prob:
                img0, img1, gt = self.cut_mix(img0, img1, gt)

            if random.uniform(0, 1) < 0.5:  # RGB to BGR
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:  # Vertical
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5:  # Horizon
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5:  # Reverse
                tmp = img1
                img1 = img0
                img0 = tmp
                if self.arbitrary == True:
                    timestep = 1- timestep

            if self.rotate90 and random.uniform(0, 1) < 0.5:  # 90 Rotate
                img0 = np.rot90(img0)
                img1 = np.rot90(img1)
                gt = np.rot90(gt)

            img0 = Image.fromarray(img0)
            img1 = Image.fromarray(img1)
            gt = Image.fromarray(gt)

            if self.rotate and random.uniform(0, 1) < 0.5:  # 0 ~ 89 Rotate
                angle = np.random.randint(0, 90)
                img0 = img0.rotate(angle)
                img1 = img1.rotate(angle)
                gt = gt.rotate(angle)

            if self.brightness > 0. and random.uniform(0, 1) < self.aug_prob:
                brightness_str = random.uniform(1 - self.brightness, 1 + self.brightness)
                img0 = F.adjust_brightness(img0, brightness_factor=brightness_str)
                img1 = F.adjust_brightness(img1, brightness_factor=brightness_str)
                gt = F.adjust_brightness(gt, brightness_factor=brightness_str)

            if self.contrast > 0. and random.uniform(0, 1) < self.aug_prob:
                contrast_str = random.uniform(1 - self.contrast, 1 + self.contrast)
                img0 = F.adjust_contrast(img0, contrast_factor=contrast_str)
                img1 = F.adjust_contrast(img1, contrast_factor=contrast_str)
                gt = F.adjust_contrast(gt, contrast_factor=contrast_str)

            if self.saturation > 0. and random.uniform(0, 1) < self.aug_prob:
                saturation_str = random.uniform(1 - self.saturation, 1 + self.contrast)
                img0 = F.adjust_saturation(img0, saturation_factor=saturation_str)
                img1 = F.adjust_saturation(img1, saturation_factor=saturation_str)
                gt = F.adjust_saturation(gt, saturation_factor=saturation_str)

            if self.hue > 0. and random.uniform(0, 1) < self.aug_prob:
                hue_str = random.uniform(-self.hue, self.hue)
                img0 = F.adjust_hue(img0, hue_factor=hue_str)
                img1 = F.adjust_hue(img1, hue_factor=hue_str)
                gt = F.adjust_hue(gt, hue_factor=hue_str)

            img0 = np.asarray(img0)
            img1 = np.asarray(img1)
            gt = np.asarray(gt)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
                
        if self.arbitrary == True:
            return torch.cat((img0, img1, gt), 0), timestep
        else:
            return torch.cat((img0, img1, gt), 0)
