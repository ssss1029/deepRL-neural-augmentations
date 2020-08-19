
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


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # The normalize code -> t.sub_(m).div_(s)
        new_tensor = torch.zeros_like(tensor)
        for i, m, s in zip(range(3), self.mean, self.std):
            new_tensor[:, i] = (tensor[:, i] - m) / s
        return new_tensor

# Useful for undoing thetorchvision.transforms.Normalize() 
# From https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # The normalize code -> t.sub_(m).div_(s)
        new_tensor = torch.zeros_like(tensor)
        for i, m, s in zip(range(3), self.mean, self.std):
            new_tensor[:, i] = (tensor[:, i] * s) + m
        return new_tensor

unnorm_fn = UnNormalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalize_fn = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def call_augfn(sample, aug_fn):
    """
    Sample: np array of unit8 of (3, H, W)
    aug_fn: expects torch tensor normalized using normalize()
    """

    assert sample.shape[0] == 3
    assert len(sample.shape) == 3

    # print(sample)

    sample = torch.from_numpy(sample).to(device=torch.device('cuda')).float() / 255.0
    sample = sample.unsqueeze(0)
    sample = normalize_fn(sample)

    with torch.no_grad():
        # print(sample)
        sample = aug_fn(sample)
        # print(sample)

    sample = unnorm_fn(sample).clamp(0, 1) * 255.0
    sample = sample.cpu().numpy().astype(np.uint8)
    sample = sample[0]

    return sample
