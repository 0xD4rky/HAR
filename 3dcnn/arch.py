import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from tqdm import tqdm


def conv3D_output_size(img_size, padding, kernel_size, stride):
    
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(torch.nn.Module):
    
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(5,64, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d( kernel_size = 2, stride =  2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size = 3, padding  = 1 )
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        
        self.conv3a = nn.Conv3d(128,256, kernel_size = 3, padding = 1)
        self.bn3a = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size = 3, padding = 1)
        self.bn3b = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size = 2, padding = 1)
        
        self.conv4a = nn.Conv3d(256,512, kernel_size = 3, padding = 1)
        self.bn4a = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size = 3, padding = 1)
        self.bn4b = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        
        self.conv5a = torch.nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn5a = torch.nn.BatchNorm3d(512)
        self.conv5b = torch.nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.bn5b = torch.nn.BatchNorm3d(512)
        self.pool5 = torch.nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.fc6 = torch.nn.Linear(512*1*7*7, 4096)
        self.fc7 = torch.nn.Linear(4096,4096)
        self.fc8 = torch.nn.Linear(4096,num_classes)
        
        self.dropout = torch.nn.Dropout(p = 0.5)
        self.relu = torch.nn.ReLU()
        
    
    def forward(self,x):
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)
        
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        x = self.pool4(x)
        
        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.relu(self.bn5b(self.conv5b(x)))
        x = self.pool5(x)
        
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        
        return x
    



        