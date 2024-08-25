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
        
def load_checkpoint(checkpoint_path, model, optimizer = None):
    """
    Load model checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"File doesn't exist {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_acc']

def save_dict_to_json(d, json_path):
    """
    Save dict of floats to json file
    """
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)