import torch
import torch.nn as nn
from model.warplayer import warp
from model.IFNet import *
from model.train_util import *
from model.refine import *
from model.arch_util import *


device = torch.device("cuda")


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MAMNet(nn.Module):
    def __init__(self, in_planes, out_planes, drop_out_rate=0.):
        super(MAMNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))
        self.norm1 = LayerNorm2d(out_planes)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=out_planes,
                               bias=True), nn.ReLU(inplace=True))
        self.sa0 = SpatialAttention()
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, out_planes, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, out_planes, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm2d(out_planes)
        self.sa1 = SpatialAttention()

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=out_planes,
                               bias=True), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))


    def forward(self, inp):
        x = self.conv1(inp)

        x_ = x

        x = self.norm1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x * self.sa0(x)
        x = self.conv4(x)

        x = self.dropout1(x)
        y = x_ + x * self.beta

        x = self.conv5(self.norm2(y))
        x = self.conv6(x)
        x = x * self.sa1(x)
        x = self.conv7(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x


class MAMNet_D(nn.Module):
    def __init__(self, in_planes, out_planes, drop_out_rate=0.):
        super(MAMNet_D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))
        self.norm1 = LayerNorm2d(out_planes)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, out_planes, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, out_planes, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm2d(out_planes)

        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=out_planes,
                               bias=True), nn.ReLU(inplace=True))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_planes // 2, out_channels=out_planes, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )


        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True), nn.ReLU(inplace=True))


    def forward(self, inp):
        x = self.conv1(inp)

        x_ = x
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.dropout1(x)
        y = x_ + x * self.beta

        x = self.conv5(self.norm2(y))
        x = self.conv6(x)
        x = x * self.se(x)
        x = self.conv7(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x

class Errormap_Net(nn.Module):
    def __init__(self):
        super(Errormap_Net, self).__init__()
        self.down0 = Conv2(9, 32)
        self.down1 = Conv2(32, 64)
        self.down2 = Conv2(64, 128)
        self.up0 = deconv(128, 64)
        self.mamd0 = MAMNet_D(64, 64)
        self.up1 = deconv(128, 32)
        self.mamd1 = MAMNet_D(32, 32)
        self.up2 = deconv(64, 8)
        self.mamd2 = MAMNet_D(8, 8)
        self.conv = nn.Conv2d(8, 3, 3, 1, 1)

    def forward(self, img_easy, img_medium, img_hard):
        s0 = self.down0(torch.cat((img_easy, img_medium, img_hard), 1))  # (b, 32, 112, 112)
        s1 = self.down1(s0)  # (b, 64, 56, 56)
        s2 = self.down2(s1)  # (b, 128, 28, 28)
        x = self.up0(s2)
        x = self.mamd0(x)
        x = self.up1(torch.cat((x, s1), 1))  # (b, 128, 28, 28)
        x = self.mamd1(x)
        x = self.up2(torch.cat((x, s0), 1))  # (b, 64, 56, 56)
        x = self.mamd2(x)
        x = self.conv(x)  # (b, 3, 224, 224)

        return torch.sigmoid(x)


class Class_VFI_intra_v2(nn.Module):
    def __init__(self):
        super(Class_VFI_intra_v2, self).__init__()
        self.net1 = IFNet()
        self.net2 = IFNet()
        self.net3 = IFNet()
        self.error_map_net = Errormap_Net()
        self.contextnet = Contextnet()
        self.synthesisnet = Unet()

    def forward(self, imgs, gt, timestep, is_train):
        if is_train:
            imgs_ = imgs
            gt_ = gt

            img0 = imgs_[:, :3, :, :]
            img1 = imgs_[:, 3:6, :, :]
        
            flow1, mask1, merged1, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta \
                = self.net1(torch.cat((imgs_, gt_), 1), scale=[4, 2, 1], timestep=timestep)

            flow2, mask2, merged2, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta \
                = self.net2(torch.cat((imgs_, gt_), 1), scale=[4, 2, 1], timestep=timestep)

            flow3, mask3, merged3, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta \
                = self.net3(torch.cat((imgs_, gt_), 1), scale=[4, 2, 1], timestep=timestep)

            easy_error_map = (((merged1[2] - gt_).abs().mean(1, True) <= (merged2[2] - gt_).abs().mean(1, True)) & ((merged1[2] - gt_).abs().mean(1, True) <= (merged3[2] - gt_).abs().mean(1, True))).float().detach()
            medium_error_map = (((merged2[2] - gt_).abs().mean(1, True) < (merged1[2] - gt_).abs().mean(1, True)) & ((merged2[2] - gt_).abs().mean(1, True) <= (merged3[2] - gt_).abs().mean(1, True))).float().detach()
            hard_error_map = (((merged3[2] - gt_).abs().mean(1, True) < (merged2[2] - gt_).abs().mean(1, True)) & ((merged3[2] - gt_).abs().mean(1, True) < (merged1[2] - gt_).abs().mean(1, True))).float().detach()

            error_map_gt = torch.cat((easy_error_map, medium_error_map, hard_error_map), 1)

            error_map_pred = self.error_map_net(merged1[2], merged2[2], merged3[2])

            out = merged1[2] * easy_error_map + merged2[2] * medium_error_map + merged3[2] * hard_error_map
            pred = merged1[2] * error_map_pred[:, :1, :, :] + merged2[2] * error_map_pred[:, 1:2, :, :] + merged3[2] * error_map_pred[:, 2:, :, :]
            flow = flow1[2] * error_map_pred[:, :1, :, :] + flow2[2] * error_map_pred[:, 1:2, :, :] + flow3[2] * error_map_pred[:, 2:, :, :]
            mask = mask1 * error_map_pred[:, :1, :, :] + mask2 * error_map_pred[:, 1:2, :, :] + mask3 * error_map_pred[:, 2:, :, :]

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        
            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            merged_no_refine = pred

            tmp = self.synthesisnet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, error_map_pred)
            res = tmp[:, :3] * 2 - 1
            pred = torch.clamp(pred + res, 0, 1)

            out_res = out
            pred_res = pred
            error_map_gt_res = error_map_gt
            error_map_pred_res = error_map_pred
            merged_no_refine_ = merged_no_refine
            flow_ = flow
            flow_teacher_ = flow_teacher
            mask_ = mask
            merged_teacher_ = merged_teacher
        else:

            imgs_ = imgs
            gt_ = gt

            img0 = imgs_[:, :3, :, :]
            img1 = imgs_[:, 3:6, :, :]
        
            flow1, mask1, merged1, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta \
                = self.net1(torch.cat((imgs_, gt_), 1), scale=[4, 2, 1], timestep=timestep)

            flow2, mask2, merged2, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta \
                = self.net2(torch.cat((imgs_, gt_), 1), scale=[4, 2, 1], timestep=timestep)

            flow3, mask3, merged3, flow_teacher, merged_teacher, loss_distill, merged_no_refine, meta \
                = self.net3(torch.cat((imgs_, gt_), 1), scale=[4, 2, 1], timestep=timestep)


            error_map_pred = self.error_map_net(merged1[2], merged2[2], merged3[2])
            pred = merged1[2] * error_map_pred[:, :1, :, :] + merged2[2] * error_map_pred[:, 1:2, :, :] + merged3[2] * error_map_pred[:, 2:, :, :]
            flow = flow1[2] * error_map_pred[:, :1, :, :] + flow2[2] * error_map_pred[:, 1:2, :, :] + flow3[2] * error_map_pred[:, 2:, :, :]
            mask = mask1 * error_map_pred[:, :1, :, :] + mask2 * error_map_pred[:, 1:2, :, :] + mask3 * error_map_pred[:, 2:, :, :]

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])

            c0 = self.contextnet(img0, flow[:, :2])
            c1 = self.contextnet(img1, flow[:, 2:4])
            merged_no_refine = pred

            tmp = self.synthesisnet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, error_map_pred)
            res = tmp[:, :3] * 2 - 1
            pred = torch.clamp(pred + res, 0, 1)
            
            out_res = pred
            pred_res = pred
            error_map_gt_res = error_map_pred
            error_map_pred_res = error_map_pred
            merged_no_refine_ = merged_no_refine
            flow_ = flow
            flow_teacher_ = flow_teacher
            mask_ = mask
            merged_teacher_ = merged_teacher

            return out_res, pred_res, flow_, mask_, flow_teacher_, merged_teacher_, loss_distill, merged_no_refine_, meta, error_map_gt_res, error_map_pred_res

        return out_res, pred_res, flow_, mask_, flow_teacher_, merged_teacher_, loss_distill, merged_no_refine_, meta, error_map_gt_res, error_map_pred_res