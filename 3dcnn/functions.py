import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from tqdm import tqdm

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