import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
import torch.nn.functional as F
from model.train_util import *
from model.refine import *
from model.classifier_intra_v2 import *

import os
import glob
# from RIFE.model.model_util import ModelEma
from collections import OrderedDict

device = torch.device("cuda")

class Model:
    def __init__(self, local_rank=-1, classifier_intra_v2=False):

        if classifier_intra_v2 == True:
            self.flownet = Class_VFI_intra_v2() 
        else:
            self.flownet = IFNet()

        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.lap = LapLoss()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, pkl, rank=0, strict=True, freeze=False):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(torch.load('{}'.format(pkl)), strict=False)
            
        if freeze:
            for name, param in self.flownet.named_parameters():
                if 'block' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        for name, param in self.flownet.named_parameters():
            print(name, ':', param.requires_grad)
        print("===============================================")

    def load_class_module(self, easy_pkl, medium_pkl, hard_pkl):
        
        self.network1 = self.flownet.net1
        self.network2 = self.flownet.net2
        self.network3 = self.flownet.net3

        self.network1.load_state_dict(torch.load(easy_pkl))

        for name, param in self.network1.named_parameters():
            param.requires_grad = False
        
        self.network2.load_state_dict(torch.load(medium_pkl))
        
        for name, param in self.network2.named_parameters():
            param.requires_grad = False
        
        self.network3.load_state_dict(torch.load(hard_pkl))
        for name, param in self.network3.named_parameters():
            param.requires_grad = False
        
        print('Loading model for branch3 [{:s}] ...'.format(hard_pkl))

        for name, param in self.flownet.named_parameters():
            print(name, ':', param.requires_grad)
        print("===============================================")

    def save_model(self, path, rank=0, epoch=0, psnr=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/flownet_{:03d}_{:.3f}.pkl'.format(path, epoch, psnr))

    def save_best_model(self, path, rank=0, epoch=0, psnr=0):
        if rank == 0:
            prev_best = glob.glob('{}/best*'.format(path))
            #print(prev_best)
            for i in range(len(prev_best)):
                os.remove(prev_best[i])
            torch.save(self.flownet.state_dict(), '{}/best_flownet_{:03d}_{:.3f}.pkl'.format(path, epoch, psnr))
    
    def save_last_model(self, path, rank=0, epoch=0, psnr=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(), '{}/last_flownet_{:03d}_{:.3f}.pkl'.format(path, epoch, psnr))


    def save_checkpoint(self, path, epoch, loss_G):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.flownet.state_dict(),
            'optimizer_state_dict': self.optimG.state_dict(),
            'loss': loss_G
        }, '{}/checkpoint.pkl'.format(path))
    
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None, timestep=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()

        flow, mask, merged, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta \
            = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1], timestep=timestep)

        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()        
        loss_G = loss_l1 + loss_tea + loss_distill * (0.01)   # when training RIFEm, the weight of loss_distill should be 0.005 or 0.002
        
        if training:
            self.optimG.zero_grad()            
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow[2]
            merged_teacher = merged[2] 
        return merged[2], {
            'merged_tea': merged_teacher,
            'merged_no_refine': merged_no_refine, # refineNet 전후 psnr 비교를 위한 출력
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[2],                # flow[2][:, :2] 에서  [:, :2] 원래 있었는데 wandb 에서 양방향 flow 보려고 없앰
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'loss_G': loss_G,
            }

    def update_classifier_intra_v2(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None, timestep=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()

        out_res, pred_res, flow, mask, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta, error_map_gt_res, error_map_pred_res = self.flownet(imgs, gt, timestep, training)
        
        loss_pred = (self.lap(merged_no_refine, out_res)).mean()
        loss_pred_gt = (self.lap(merged_no_refine, gt)).mean()
        loss_error_map1 = nn.L1Loss()(error_map_pred_res[:, :1, :, :], error_map_gt_res[:, :1, :, :])
        loss_error_map2 = nn.L1Loss()(error_map_pred_res[:, 1:2, :, :], error_map_gt_res[:, 1:2, :, :])
        loss_error_map3 = nn.L1Loss()(error_map_pred_res[:, 2:3, :, :], error_map_gt_res[:, 2:3, :, :])
        loss_l1 = (self.lap(pred_res, gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        
        loss_G = 0.5 * loss_pred_gt + 0.5 * loss_l1
        
        if training:
            self.optimG.zero_grad()            
            loss_G.backward()
            self.optimG.step()
        else:
            flow_teacher = flow
            merged_teacher = pred_res 
        
        return pred_res, {
            'merged_tea': merged_teacher,
            'merged_no_refine': merged_no_refine, # refineNet 전후 psnr 비교를 위한 출력
            'mask': mask,
            'mask_tea': mask,
            'flow': flow,                # flow[2][:, :2] 에서  [:, :2] 원래 있었는데 wandb 에서 양방향 flow 보려고 없앰
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            'loss_G': loss_G,
            'loss_pred' : loss_pred,
            'loss_pred_gt': loss_pred_gt,
            'loss_errormap1' : loss_error_map1,
            'loss_errormap2' : loss_error_map2,
            'loss_errormap3' : loss_error_map3,
            }
    
    def ema_update(self):
        self.model_ema.update(self.flownet)

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5):
        imgs = torch.cat((img0, img1), 1)
        training = False
        if self.classifier == True:
            out_res, type_res, flow, mask, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta = self.flownet(imgs, imgs, timestep=timestep, is_train=training)
        else:
            flow, mask, merged, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta = self.flownet(imgs, scale_list, timestep=timestep)

        if TTA == False:
            if self.classifier == True:
                return out_res
            else:
                return merged[2]
        else:
            flow2, mask2, merged2, flow_teacher2, merged_teacher2, loss_distill2, merged_no_refine = self.flownet(imgs.flip(2).flip(3), scale_list, timestep=timestep)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
