import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

set_seed(42)

data_path = r'C:\Users\DELL\HAR\data'
train_path = os.path.join(data_path,"train")
test_path = os.path.join(data_path,"test")
val_path = os.path.join(data_path,"val")


frame_path = r'C:\Users\DELL\HAR\static\frame_path'
optical_flow_path = r'C:\Users\DELL\HAR\static\optical_flow'

train_csv = os.path.join(data_path,"train.csv")
test_csv = os.path.join(data_path,"test.csv")
val_csv = os.path.join(data_path,"val.csv")


def create_directories(base_path, class_name, video_name):

    """creating and storing frame paths

    Returns:
        string: returns created video paths where frames are supposed to be stored
    """
    class_path = os.path.join(base_path,class_name)
    os.makedirs(class_path, exist_ok = True) 
    video_path = os.path.join(class_path,video_name)
    os.makedirs(video_path, exist_ok = True)
    
    return video_path

def save_frames(frame, class_name, video_name, frame_number):
    
    """
    saves frames in created video paths
    """

    video_path = create_directories(frame_path, class_name, video_name)
    frame_file_path = os.path.join(video_path, f"frame_{frame_number:04d}.jpg")
    cv2.imwrite(frame_file_path,frame)
    
def save_optical_flow(flow, class_name, video_name, frame_number):
    """
    saves optical flows in created flow paths
    """
    
    flow_path = create_directories(optical_flow_path,class_name,video_name)
    flow_file_path = os.path.join(flow_path, f"flow_{frame_number:04d}.jpg")
    cv2.imwrite(flow_file_path,flow)
    
os.makedirs(frame_path, exist_ok=True)
os.makedirs(optical_flow_path, exist_ok=True)   

# some more params:
frame_size = (224,224)
clip_length = 16
num_classes = 101

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(frame_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_frames(video_path, frame_path):
    
    """
    extracting frames to frame_path

    Returns:
        int: frame count i.e. total extracted frames
    """
    
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size)
        cv2.imwrite(os.path.join(frame_path, f"frame_{frame_count:04d}.jpg"), frame)
        frame_count += 1
    
    video.release()
    return frame_count


def compute_optical_flow(prev_frame,curr_frame):
    
    """computing optical flow of current and the previous frames

    Returns:
        numpy array: return computed optical flow
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) # converting flow to RGB
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return flow_rgb


def process_video(video_path, frame_path, optical_flow_path):
    video_name = os.path.basename(video_path)
    frame_path = os.path.join(frame_path, video_name.split('.')[0])
    optical_flow_path = os.path.join(optical_flow_path, video_name.split('.')[0])
    
    os.makedirs(frame_path, exist_ok=True)
    os.makedirs(optical_flow_path, exist_ok=True)
    
    frame_count = extract_frames(video_path, frame_path)
    
    for i in range(1, frame_count):
        prev_frame = cv2.imread(os.path.join(frame_path, f"frame_{i-1:04d}.jpg"))
        curr_frame = cv2.imread(os.path.join(frame_path, f"frame_{i:04d}.jpg"))
        
        flow = compute_optical_flow(prev_frame, curr_frame)
        cv2.imwrite(os.path.join(optical_flow_path, f"flow_{i:04d}.jpg"), flow)
    
def process_dataset(data_path, frame_path, optical_flow_path):
    for class_name in tqdm(os.listdir(data_path), desc="Processing classes"):
        class_path = os.path.join(data_path, class_name)
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            process_video(video_path, os.path.join(frame_path, class_name), os.path.join(optical_flow_path, class_name))


print("Processing train dataset...")
process_dataset(train_path, os.path.join(frame_path, "train"), os.path.join(optical_flow_path, "train"))
print("Processing test dataset...")
process_dataset(test_path, os.path.join(frame_path, "test"), os.path.join(optical_flow_path, "test"))
print("Processing val dataset")
process_dataset(val_path, os.path.join(frame_path, "val"), os.path.join(optical_flow_path, "val"))

print("Video processing completed.")


        
        
        
        
    

