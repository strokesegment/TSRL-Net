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


from net import *
from data import *
from loss import *





if __name__ == "__main__":
    
    model_save_path = '....pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # add
    print(device)
    
    train_dataset = Traindata_Loader()
    print("the total num of train data:", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=32,num_workers=4,shuffle=True)
    net = tsrl_net(in_channels=4).to(device)
    criterion = target_loss(patch_size = 4, alpha = 0.25 * 3).to(device)
    optimizer = optim.Adam(net.parameters(),weight_decay=0.00001)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size = 10,gamma = 0.96)
    
    
    for epoch in range(150):
        net.train()
        for step, (x, y , z) in enumerate(train_loader):
            
            x = torch.as_tensor(x, dtype=torch.float32).to(device)
            y = torch.as_tensor(y, dtype=torch.float32).to(device)
            z = torch.as_tensor(z, dtype=torch.float32).to(device)
            optimizer.zero_grad()
            output,out4,out3,out2,out1,we1,we2,we3,we4,we5 = net(x)

            # 损失函数计算
            loss = criterion(output, y)
            # 多层监督损失函数
            loss0 = criterion(out4, y)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            loss1 = criterion(out3, y)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            loss2 = criterion(out2, y)
            y = F.interpolate(y, scale_factor=0.5, mode='bilinear', align_corners=True)
            loss3 = criterion(out1, y)
            # 注意力损失函数
            loss_1 = criterion_attention(we1, z)
            loss_2 = criterion_attention(we2, z)
            loss_3 = criterion_attention(we3, z)
            loss_4 = criterion_attention(we4, z)
            loss_5 = criterion_attention(we5, z)
            loss_total = loss + loss0 + loss1 + loss2 + loss3 + 0.15*(loss_1 + loss_2 + loss_3 + loss_4) + 0.5 * loss_5
            iter_loss = loss.item()
            loss_total.backward()
            optimizer.step()

        #scheduler.step()

        model_path=model_save_path #+ str(epoch) +'.pth'
        torch.save(net.state_dict(), model_path)

