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





class SobelOperator(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = torch.tensor(x_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_x.weight.requires_grad = False

        y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 4
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight.data = torch.tensor(y_kernel).unsqueeze(0).unsqueeze(0).float().cuda()
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b * c, 1, h, w)

        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x



class enhance_mixing_loss(nn.Module):
    def __init__(self, patch_size = 2, alpha = 0.75):
        super(enhance_mixing_loss, self).__init__()
        self.sobel = SobelOperator(1e-4)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=patch_size)
        self.up = nn.Upsample(scale_factor=patch_size, mode='nearest')
        self.alpha = alpha
    def forward(self, y_pred, y_true):
        gamma = 2.0
        smooth = 1.
        epsilon = 1e-7

        max_map = self.MaxPool2d(y_true)
        up_map = self.up(max_map)
        alpha_map = torch.where(torch.eq(up_map, 1), torch.ones_like(y_pred) * self.alpha, torch.ones_like(y_pred) * (1 - self.alpha))

        #sobel_p = self.sobel(y_pred)
        #sobel_g = self.sobel(y_true)

        # y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        alpha_map = alpha_map.view(-1)
        print(alpha_map)

        # dice_loss
        intersection = (y_true * y_pred).sum()
        dice_loss = (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)

        # focal_loss
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)  # 最小值时1e-7
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
        focal_loss = -torch.mean(self.alpha * torch.pow(1.0 - pt_1, gamma) * torch.log(pt_1)) - torch.mean(
            alpha_map * torch.pow(pt_0, gamma) * torch.log(1.0 - pt_0))
        #sobel_loss = F.l1_loss(sobel_p, sobel_g)
        return focal_loss - torch.log(dice_loss) #+  sobel_loss



class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()
       
    def forward(self, y_pred, y_true):
        gamma = 2.0
        smooth = 1.
        epsilon = 1e-7
        
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        
        intersection = (y_true * y_pred).sum()
        dice_loss = (2. * intersection + smooth) / ((y_true * y_true).sum() + (y_pred * y_pred).sum() + smooth)

        return  - torch.log(dice_loss) 


class target_aware_loss(nn.Module):
    def __init__(self, patch_size = 2, alpha = 0.75):
        super(target_loss, self).__init__()
        self.sobel = SobelOperator(1e-4)
        self.MaxPool2d = nn.MaxPool2d(kernel_size=patch_size)
        self.up = nn.Upsample(scale_factor=patch_size, mode='nearest')
        self.alpha = alpha


    def forward(self, y_pred, y_true):
        gamma = 2.0
        smooth = 1.
        epsilon = 1e-7

        max_map = self.MaxPool2d(y_true)
        up_map = self.up(max_map)
        alpha_map = torch.where(torch.eq(up_map, 1), torch.ones_like(y_pred) * self.alpha, torch.ones_like(y_pred) * (1 - self.alpha))

        
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        alpha_map = alpha_map.view(-1)
       
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon) 
        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
        focal_loss = -torch.mean(self.alpha * torch.pow(1.0 - pt_1, gamma) * torch.log(pt_1)) - torch.mean(
            alpha_map * torch.pow(pt_0, gamma) * torch.log(1.0 - pt_0))
       
        return focal_loss 



class focal_loss(nn.Module):
    def __init__(self):
        super(focal_loss, self).__init__()
        

    def forward(self, y_pred, y_true):
        gamma = 1.1
        alpha = 0.48
        smooth = 1.
        epsilon = 1e-7
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

 
        y_pred = torch.clamp(y_pred, epsilon)

        pt_1 = torch.where(torch.eq(y_true, 1), y_pred, torch.ones_like(y_pred))
        pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))
        focal_loss = -torch.mean(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - \
                 torch.mean((1 - alpha) * torch.pow(pt_0, gamma) * torch.log(1. - pt_0))
        return focal_loss 

class bce_loss(nn.Module):
    def __init__(self):
        super(bce_loss, self).__init__()
        self.criterion = nn.BCELoss(weight=None, size_average=True)
        
    def forward(self, y_pred, y_true):
        smooth = 1.
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1-epsilon)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        # dice_loss

       
        # binary_loss
        binary_loss = self.criterion(y_pred,y_true)

        return binary_loss

class Balanced_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        loss = 0.0
        # version2
        for i in range(input.shape[0]):
            beta = 1-torch.sum(target[i])/target.shape[1]
            x = torch.max(torch.log(input[i]), torch.tensor([-100.0]).cuda())
            y = torch.max(torch.log(1-input[i]), torch.tensor([-100.0]).cuda())
            l = -(beta*target[i] * x + (1-beta)*(1 - target[i]) * y)
            loss += torch.sum(l)
        return loss


class bce_dice_loss(nn.Module):
    def __init__(self):
        super(bce_dice_loss, self).__init__()
        self.criterion = nn.BCELoss(weight=None, size_average=True)
        
    def forward(self, y_pred, y_true):
        smooth = 1.
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1-epsilon)
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        # dice_loss

        intersection = (y_true * y_pred).sum()
        dice_loss = (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)
        dice_loss=1.0 - dice_loss
        # binary_loss
        binary_loss = self.criterion(y_pred,y_true)

        return binary_loss + dice_loss



class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        ALPHA = 0.25
        BETA = 0.75     
        alpha=ALPHA
        beta=BETA
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky