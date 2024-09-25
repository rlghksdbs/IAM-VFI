import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvPRelU(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(ConvPRelU, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv2(x)
        x = self.prelu(x)
        return x
    
c = 16
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()

        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)

        '''
        self.conv1 = Conv2PRelU(3, c)
        self.conv2 = Conv2PRelU(c, 2*c)
        self.conv3 = Conv2PRelU(2*c, 4*c)
        self.conv4 = Conv2PRelU(4*c, 8*c)
        '''

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)  # 16 channel
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)  # 32 channel
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)   # 64 channel
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)  # 128 channel
        return [f1, f2, f3, f4]

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(20, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, error_map_pred):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow, error_map_pred), 1))  # (b, 32, 112, 112)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))  # (b, 64, 56, 56)
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))  # (b, 128, 28, 28)
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))  # (b, 256, 14, 14)
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))  # (b, 128, 28, 28)
        x = self.up1(torch.cat((x, s2), 1))  # (b, 64, 56, 56)
        x = self.up2(torch.cat((x, s1), 1))  # (b, 32, 112, 112)
        x = self.up3(torch.cat((x, s0), 1))  # (b, 16, 224, 224)
        x = self.conv(x)  # (b, 3, 224, 224)

        return torch.sigmoid(x)

