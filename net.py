import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.optim import lr_scheduler
import sys
import h5py
import random
import copy
from matplotlib import pyplot as plt
from PIL import Image
import time
from torch.utils.data import Dataset,random_split
from torch import optim
from time import time
from torchvision import transforms
import glob
import math
import xlwt
import xlrd                           #导入模块
from xlutils.copy import copy
import torch
from torch import nn
from torch.nn import functional as F
import random


class Expand(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.unsqueeze(x, dim=0)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.squeeze(x, dim=1)


class SE_block(nn.Module):

    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.avg_2d = F.adaptive_avg_pool2d
        self.dense_block_2d = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // ratio, bias=False),
            nn.PReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.dense_block_3d = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // ratio, bias=False),
            nn.PReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, in_2d, in_3d):
        av_2d = self.avg_2d(in_2d, (1, 1))
        av_3d = self.avg_2d(in_3d, (1, 1))
        total_features = torch.cat((av_2d, av_3d), dim=1)
        filters = total_features.size(1)
        reshape_size = (total_features.size(0), 1, 1, filters)
        se = torch.reshape(total_features, reshape_size)
        se_2d = self.dense_block_2d(se)
        se_2d = se_2d.permute(0, 3, 1, 2)
        out_2d = in_2d * se_2d

        se_3d = self.dense_block_3d(se)
        se_3d = se_3d.permute(0, 3, 1, 2)
        out_3d = in_3d * se_3d
        return out_2d, out_3d

class CRL(nn.Module):
    def __init__(self):
        super(CRL, self).__init__()

    def forward(self, x, pred,ca):

        residual = x
        att = 1 - pred
        att_x = x * att
        att_x = att_x + ca
        out = residual * att_x

       # att = 1 - pred
        #out = x * x * att + x * ca + x

        return out



class Patch_attention(nn.Module):
    def __init__(self,feature_size):
        super().__init__()
        self.AvgPool2d = nn.AvgPool2d(kernel_size=int(feature_size/6))
        self.dense_block = nn.Sequential(
            nn.Linear(36, 18, bias=False),
            nn.PReLU(),
            nn.Linear(18, 36, bias=False),
            nn.Sigmoid(),
            )
        self.up=nn.Upsample(scale_factor=int(feature_size/6), mode='nearest')
    def forward(self,x):
    ##  先做通道上的全局平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        border_width=avg_out.shape[2]
        channel_num=avg_out.shape[0]
    ## 做特征图的池化，提取每一块的全局信息
        Avg_res = self.AvgPool2d(avg_out)
        reshape_size = (channel_num, 1, 1, 36)
        Avg_res = torch.reshape(Avg_res, reshape_size)
    ##  FC计算每一个patch的权重，然后转换成图像形状的权重
        linear_weight=self.dense_block(Avg_res)
        photo_weight=linear_weight.reshape(channel_num, 1, 6, 6)
    #print(photo_weight)
        end_weight=self.up(photo_weight)
        out=x+x*end_weight
        return out , torch.squeeze(linear_weight)


class BN_block2d_e(nn.Module):
    """
        2-d batch-norm block
    """

    def __init__(self, in_channels, out_channels,dropout_p,fea_size):
        super().__init__()
        self.bn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.attention = Patch_attention(fea_size)
    def forward(self, x):
        return self.attention(self.bn_block(x))



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BN_block2d(nn.Module):
    """
        2-d batch-norm block
    """

    def __init__(self, in_channels, out_channels,dropout_p):
        super().__init__()
        self.bn_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.bn_block(x)


class BN_block3d(nn.Module):
    """
        3-d batch-norm block
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.bn_block(x)


class D_SE_Add(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.SE_block = SE_block(in_channels)
        self.squeeze_block_3d = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1, padding=0),
            Squeeze(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, in_3d, in_2d):
        in_3d = self.squeeze_block_3d(in_3d)
        se_2d, se_3d = self.SE_block(in_2d, in_3d)
        out = se_2d + se_3d

        return out




class up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_block, self).__init__()
        self.Up = nn.Sequential(
        nn.ConvTranspose2d(in_channels,in_channels,3,2,1,1,1),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
        self.conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
        #self.non_local = NonLocalBlock(out_channels)
    def forward(self, x):
        asp=self.Up(x)
        out=self.conv(asp)
        #out = self.non_local(out)
        return out



def weights_init_he(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x



class DUnet(nn.Module):
    def __init__(self, in_channels, weights_init=True):
        super().__init__()

        self.in_channels = in_channels
        in_channels_3d = 1

        self.Expand = Expand
        self.MaxPool3d = nn.MaxPool3d(kernel_size=2)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2)
        self.Dropout = nn.Dropout(0.3)

        # 3d down
        self.bn_3d_1 = BN_block3d(in_channels_3d, in_channels_3d * 32)
        self.bn_3d_2 = BN_block3d(in_channels_3d * 32, in_channels_3d * 64)
        self.bn_3d_3 = BN_block3d(in_channels_3d * 64, in_channels_3d * 128)

        # 2d down

        self.bn_2d_1 = BN_block2d_e(in_channels, in_channels * 8,0.0,192)
        
        
        self.bn_2d_2 = BN_block2d_e(in_channels * 8, in_channels * 16,0.2,96)
        self.se_add_2 = D_SE_Add(in_channels * 16, in_channels * 16, 2)
        
        
        self.bn_2d_3 = BN_block2d_e(in_channels * 16, in_channels * 32,0.3,48)
        self.se_add_3 = D_SE_Add(in_channels * 32, in_channels * 32, 1)
        
        self.bn_2d_4 = BN_block2d_e(in_channels * 32, in_channels * 64,0.4,24)
        
        self.bn_2d_5 = BN_block2d_e(in_channels * 64, in_channels * 128,0.5,12)
        
        

        self.crl1 = CRL()
        self.crl2 = CRL()
        self.crl3 = CRL()
        self.crl4 = CRL()

        self.up1 = nn.Upsample(scale_factor=int(24 / 6), mode='nearest')
        self.up2 = nn.Upsample(scale_factor=int(48 / 6), mode='nearest')
        self.up3 = nn.Upsample(scale_factor=int(96 / 6), mode='nearest')
        self.up4 = nn.Upsample(scale_factor=int(192 / 6), mode='nearest')
       


        self.up_block_1 = up_block(in_channels * 128, in_channels * 64)
        self.bn_2d_6 = BN_block2d(in_channels * 128, in_channels * 64,0.5)
        self.sideout0 = SideoutBlock(in_channels * 64, 1)

        self.up_block_2 = up_block(in_channels * 64, in_channels * 32)
        self.bn_2d_7 = BN_block2d(in_channels * 64, in_channels * 32,0.4)
        self.sideout1 = SideoutBlock(in_channels * 32, 1)

        self.up_block_3 = up_block(in_channels * 32, in_channels * 16)
        self.bn_2d_8 = BN_block2d(in_channels * 32, in_channels * 16,0.2)
        self.sideout2 = SideoutBlock(in_channels * 16, 1)

        self.up_block_4 = up_block(in_channels * 16, in_channels * 8)
        self.sideout3 = SideoutBlock(in_channels * 8, 1)
        self.bn_2d_9 = BN_block2d(in_channels * 16, in_channels * 8,0.0)
        

        self.conv_10 = nn.Sequential(
            nn.Conv2d(in_channels * 8, 1 , kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # He initialization stated in the original paper
        if weights_init:
            self.apply(weights_init_he)

    def forward(self, x):
        bs_size = x.shape[0]
        input3d = self.Expand()(x)  # 1, batch_size, 4, 192, 192
        input3d = input3d.permute(1, 0, 2, 3, 4)  # batch_size, 1, 4, 192, 192

        # 3d Stream
        conv3d1 = self.bn_3d_1(input3d)
        pool3d1 = self.MaxPool3d(conv3d1)

        conv3d2 = self.bn_3d_2(pool3d1)
        pool3d2 = self.MaxPool3d(conv3d2)

        conv3d3 = self.bn_3d_3(pool3d2)

        # 2d Encoding
        in_channels = self.in_channels

        conv1,we1 = self.bn_2d_1(x)
        pool1 = self.MaxPool2d(conv1)

        conv2,we2 = self.bn_2d_2(pool1)
        conv2 = self.se_add_2(conv3d2, conv2)
        pool2 = self.MaxPool2d(conv2)

        conv3,we3 = self.bn_2d_3(pool2)
        conv3 = self.se_add_3(conv3d3, conv3)
        pool3 = self.MaxPool2d(conv3)

        conv4,we4 = self.bn_2d_4(pool3)
        pool4 = self.MaxPool2d(conv4)

        conv5,we5 = self.bn_2d_5(pool4)

        c_weight = we5.reshape(bs_size, 1, 6, 6)
        up6 = self.up_block_1(conv5)

        out1 = self.sideout0(up6)
        decoder_we1=self.up1(c_weight)
        merge6 = torch.cat(([self.crl1(conv4,out1,decoder_we1), up6]), 1)
        conv6 = self.bn_2d_6(merge6)
        

        up7 = self.up_block_2(conv6)
        out2 = self.sideout1(up7)
        decoder_we2=self.up2(c_weight)
        merge7 = torch.cat(([self.crl2(conv3,out2,decoder_we2), up7]), 1)
        conv7 = self.bn_2d_7(merge7)
        

        up8 = self.up_block_3(conv7)   #96
        out3 = self.sideout2(up8)
        decoder_we3=self.up3(c_weight)
        merge8 = torch.cat(([self.crl3(conv2,out3,decoder_we3), up8]), 1)
        conv8 = self.bn_2d_8(merge8)
        

        up9 = self.up_block_4(conv8)
        out4 = self.sideout3(up9)
        decoder_we4=self.up4(c_weight)
        merge9 = torch.cat(([self.crl4(conv1,out4,decoder_we4), up9]), 1)
        conv9 = self.bn_2d_9(merge9)
        
        conv10 = self.conv_10(conv9)

        return conv10,out4,out3,out2,out1,we1,we2,we3,we4,we5