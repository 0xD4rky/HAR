import json, os, math
import matplotlib.pyplot as pl
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from arch import CNN3D

def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir):
    """
    saving model checkpoint and best model
    """
    
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(best_model_dir, 'best_model.pth')
        torch.save(state['state_dict'], best_model_path)
        
