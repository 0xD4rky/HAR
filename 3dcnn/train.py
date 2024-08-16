import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

def train(log_interval, model, epochs, optimizer, device, train_loader):
    
    model.train()
    losses = []
    scores = []
    N_count = 0 # total trained sample in one epoch
        
    for batch_idx, (X,y) in enumerate(train_loader): # train-loader -> {batch, x,y}
        
        X,y = X.to(device), y.to(device).view(-1,)
        
        N_count = X.size(0)
        optimizer.zero_grad()
        outputs = model(X)
        loss = F.cross_entropy(outputs,y)
        losses.append(loss.item())
        
        y_pred = torch.max(outputs,1)[1]
        step_score = accuracy_score(y.cpu().data().squeeze().numpy(), y_pred.cpu().data().squeeze().numpy())
        scores.append(step_score)
        
        loss.backward()
        optimizer.step()
        
        #printing info
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epochs + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses,scores


def validation(model,test_loader,epochs,optimizer):
    
    model.eval()
    
    test_loss = 0
    all_y = []
    all_ypred = []

    with torch.no_grad():
        
        for X,y in test_loader:
            
            outputs = model(X)
            loss = F.cross_entropy(outputs,y, reduction = 'sum')
            test_loss = l