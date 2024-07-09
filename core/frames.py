import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from imutils import paths

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
    
import shutil
import imageio
import cv2

import sys
from tqdm import tqdm
import time
import copy

# Set up paths
Sequence_length=5
IMAGE_HEIGHT , IMAGE_WIDTH= 224, 224
data_dir = "/content/drive/MyDrive/Dataset/Dataset"
frames_dir = "/content/drive/MyDrive/Dataset/Model_Save"

# Get class names and their corresponding directories
classes_dir = data_dir
class_names = os.listdir(classes_dir)
class_dirs = [os.path.join(classes_dir, c) for c in class_names]

# Loop over each class directory
for i, class_dir in enumerate(class_dirs):
    # Create a directory to store the frames
    frames_class_dir = os.path.join(frames_dir, class_names[i])
    os.makedirs(frames_class_dir, exist_ok=True)

    # Loop over each video file in the class directory
    for video_file in os.listdir(class_dir):
        video_path = os.path.join(class_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        # Create a directory to store the frames for this video
        frames_video_dir = os.path.join(frames_class_dir, video_name)
       # os.makedirs(frames_video_dir, exist_ok=True)

        # Read the video and extract its frames
        reader = cv2.VideoCapture(video_path)
        video_frames_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(video_frames_count/Sequence_length), 1)
        
        for frame_counter in range(Sequence_length):
            
        # Set the current frame position of the video.
            reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

            # Reading the frame from the video. 
            success, frame = reader.read() 

            # Check if Video frame is not successfully read then break the loop
            if not success:
                break

            # Resize the Frame to fixed height and width.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            frame_path = os.path.join(frames_class_dir, f"{frame_counter * skip_frames_window}.jpg")
            imageio.imwrite(frame_path, resized_frame)
          
        reader.release()


print("Frames extraction completed successfully!")