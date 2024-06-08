import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
from utils import filename2id

frames_path = "../data/frames"
features_path = "../data/features"
test_frames_path = "../data/frames_test"
test_features_path = "../data/features_test"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet101(pretrained=True)
        self.model.fc = nn.Identity()  # Remove the classification layer

    def forward(self, x):
        return self.model(x)

# Initialize the model
feature_extractor = FeatureExtractor()
feature_extractor.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    feature_extractor.to(device)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.cpu().squeeze().numpy()

class OpticalFlowExtractor:
    def __init__(self, frame, picture_size=(224, 224)):
        self.model = cv2.optflow.createOptFlow_DeepFlow()
        self.picture_size = picture_size
        self.frame1 = cv2.resize(frame, picture_size)
    
    def extract(self, frame2):
        frame2 = cv2.resize(frame2, self.picture_size)
        ret = self.model.calc(self.frame1, frame2, None)
        self.frame1 = frame2
        return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preparation')
    parser.add_argument('--training', action='store_true', help='For model training')

    args = parser.parse_args()
    training_mode = args.training
    
    if training_mode:
        for i in range(1, 7):
            frames_dir = os.path.join(frames_path, f"{i:02d}")
            features_dir = os.path.join(features_path, f"{i:02d}")
            os.makedirs(features_dir, exist_ok=True)

            # List all frames
            frames = os.listdir(frames_dir)
            frames.sort(key=lambda x: filename2id(x))
            
            # initialize optical flow extractor
            pic1 = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, frames[0])), cv2.COLOR_BGR2GRAY)
            optical_flow_extractor = OpticalFlowExtractor(pic1)

            # Use tqdm to display progress bar
            for frame in tqdm(frames, desc=f'Extracting features for video {i:02d}', unit='frame'):
                frame_path = os.path.join(frames_dir, frame)
                features = extract_features(frame_path)
                fp = os.path.join(features_dir, frame.replace('.jpg', '.npy'))
                np.save(fp, features)
                pic2 = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
                flow = optical_flow_extractor.extract(pic2) # np array of (224, 224, 2)
                fp = os.path.join(features_dir, frame.replace('.jpg', '_flow.npy'))
                np.save(fp, flow)
            
    else:
        for i in range(1, 10, 1):
            frames_dir = os.path.join(test_frames_path, f"{i:02d}")
            features_dir = os.path.join(test_features_path, f"{i:02d}")
            os.makedirs(features_dir, exist_ok=True)

            # List all frames
            frames = os.listdir(frames_dir)
            
            # initialize optical flow extractor
            pic1 = cv2.cvtColor(cv2.imread(os.path.join(frames_dir, frames[0])), cv2.COLOR_BGR2GRAY)
            optical_flow_extractor = OpticalFlowExtractor(pic1)

            # Use tqdm to display progress bar
            for frame in tqdm(frames, desc=f'Extracting features for video {i:02d}', unit='frame'):
                frame_path = os.path.join(frames_dir, frame)
                features = extract_features(frame_path)
                fp = os.path.join(features_dir, frame.replace('.jpg', '.npy'))
                np.save(fp, features)
                pic2 = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
                flow = optical_flow_extractor.extract(pic2) # np array of (224, 224, 2)
                fp = os.path.join(features_dir, frame.replace('.jpg', '_flow.npy'))
                np.save(fp, flow)
