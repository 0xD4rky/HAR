import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

from utils import save_checkpoint, load_checkpoint, save_dict_to_json
from arch import CNN3D
from functions import *


class Trainer:
    
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, device, config):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.config = config
    
    def train_epoch(self):
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc = 'Training', leave = False)
        for inputs, labels in pbar:
            
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs,labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss = loss.item()*inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': loss.item(), 'Acc': 100. * correct / total})
            
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validating', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'Loss': loss.item(), 'Acc': 100. * correct / total})
        
        epoch_loss = running_loss / len(self.val_dataloader.dataset)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train_and_evaluate(self):
        best_val_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            is_best = val_acc > best_val_acc
            best_val_acc = max(val_acc, best_val_acc)
            
            # Save checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc': best_val_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, self.config.checkpoint_dir, self.config.best_model_dir)
            
            # Save training metrics
            save_dict_to_json({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(self.config.checkpoint_dir, f'metrics_epoch_{epoch+1}.json'))
            
            print()