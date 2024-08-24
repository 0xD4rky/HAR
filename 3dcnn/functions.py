import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils import data
from tqdm import tqdm
import pandas as pd
import cv2

# --------------------- LABEL CONVERSION TOOLS --------------------- #
def labels2cat(label_encoder,list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder,label_encoder,list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1,1)).toarray()

def onehot2labels(label_encoder,y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder,y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

## ---------------------- Dataloaders ---------------------- ##

class Dataset_3DCNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        folder = self.folders[index]

        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        return X, y
    
    
## -------------------- (reload) model prediction ---------------------- ##
def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1] 
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred


## ------------------ DATASET AND DATALOADERS ------------------------##

class UCF101Dataset(Dataset):
    
    def __init__(self, data_path, frame_path, optical_flow_path, csv_file, clip_length = 16):
        
        self.data_path = data_path
        self.frame_path = frame_path
        self.optical_flow_path = optical_flow_path
        self.clip_length = clip_length
        
        self.df = pd.read_csv(csv_file)
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        
        row = self.df.iloc(idx)
        video_path = os.path.join(self.data_path, row['class'], row['video'])
        frame_dir = os.path.join(self.frame_path, row['class'], row['video'].split('.')[0])
        flow_dir = os.path.join(self.optical_flow_path, row['class'], row['video'].split('.')[0])
        
        num_frames = len([f for f in os.listdir(frame_dir) if f.endswith ('.jpg')])
        
        start_frame = np.random.randint(0, max(1, num_frames - self.clip_length))
        
        frames = []
        flows = []
        
        for i in range(start_frame, min(start_frame + self.clip_length, num_frames)):
            frame = cv2.imread(os.path.join(frame_dir, f"frame_{i:04d}.jpg"))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            if i > start_frame:
                
                flow = cv2.imread(os.path.join(flow_dir, f"flow_{i:04d}.jpg"))
                flow = cv2.cvtColor(flow, cv2.COLOR_BGR2RGB)
                flows.append(flow)
                
        
        while len(frames) < self.clip_length:
            
            frames.append(torch.zeros_like(frames[0]))
        while len(flows) < self.clip_length - 1:
            flows.append(torch.zeros_like(flows[0]))
            
        frames = torch.stack(frames)
        flows = torch.stack(flows)
        
        clip = torch.cat([frames,flows], dim = 0)

        class_idx = self.class_to_idx[row['label']]
        
        return clip, class_idx

