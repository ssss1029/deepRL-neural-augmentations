import sys
import os
import numpy as np
import os
import shutil
import tempfile
from PIL import Image
import random
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms as trn
from torchvision import datasets
import torchvision.transforms.functional as trnF 
from torch.nn.functional import gelu, conv2d
import torch.nn.functional as F
import random
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class GELU(torch.nn.Module):
    def forward(self, x):
        return F.gelu(x)

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, hidden_planes=24, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = hidden_planes
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale -1
        
        convs = []
        bns = []
        for i in range(self.nums):
            K = random.choice([1, 3, 5])
            D = random.choice([1, 2, 3])
            P = int(((K - 1) / 2) * D)

            convs.append(nn.Conv2d(width, width, kernel_size=K, stride = stride, padding=P, dilation=D, bias=True))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i==0 or self.stype=='stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.act(self.bns[i](sp))
          
            if i==0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1 and self.stype=='normal':
            out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
            out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        # out = self.act(out)

        return out

class Res2Net(torch.nn.Module):
    def __init__(self, epsilon=0.2, hidden_planes=24):
        super(Res2Net, self).__init__()
        
        self.epsilon = epsilon
        self.hidden_planes = hidden_planes
                
        self.block1 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        self.block2 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        self.block3 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        self.block4 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        # self.block5 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
        # self.block6 = Bottle2neck(3, 3, hidden_planes=hidden_planes)
    
    def forward_original(self, x):
                
        x = (self.block1(x) * self.epsilon) + x
        x = (self.block2(x) * self.epsilon) + x
        # x = (self.block3(x) * self.epsilon) + x
        # x = (self.block4(x) * self.epsilon) + x
        
        # if random.random() < 0.5:
        #     x = (self.block5(x) * self.epsilon) + x
        
        # if random.random() < 0.5:
        #     x = (self.block6(x) * self.epsilon) + x
        
        return x

    def forward_randorder(self, x):
        
        num_splits = random.choice([2, 3, 6])
        # print("num_splits = ", num_splits)
        per_split = 6 / num_splits
        # blocks = [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]
        blocks = [self.block1, self.block2, self.block3, self.block4]
        random.shuffle(blocks)
        
        split_blocks = [blocks[int(round(per_split * i)): int(round(per_split * (i + 1)))] for i in range(num_splits)]
        
        for group in split_blocks:
            group_len = len(group)
            
            branch = x
            for block in group:
                branch = block(branch) * self.epsilon
                branch = branch + ((torch.rand_like(branch) - 0.5) * random.random() * 0.5)
            x = x + branch 
                
        return x
    
    def eval_random_block(self, x):
        # blocks = [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6]
        blocks = [self.block1, self.block2, self.block3, self.block4]
        block = random.choice(blocks)
        return block(x)
    
    def forward_multisplit(self, x):
        
        for section in range(6):
            
            splits = random.choice([1,2,3])
            blocks = random.choice([1,2,3])
            split_output = torch.zeros_like(x)
            
            for split in range(splits):
                branch = x.clone()
                for block in range(blocks):
                    branch = self.eval_random_block(x)
                
                split_output = split_output + (branch * self.epsilon / splits)
            
            x = split_output + x
            
        return x

    def forward(self, x):
        # funcs = [
        #     self.forward_original,
        #     self.forward_multisplit,
        #     self.forward_randorder
        # ]
        
        # random.shuffle(funcs)
        
        # F1 = funcs[0]
        # F2 = funcs[1]
        # F3 = funcs[2]
        # return F1(x)

        return self.forward_original(x)
