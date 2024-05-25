import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.resnet50(pretrained=True)
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
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(image)
    return features.squeeze().numpy()

if __name__ == "__main__":
    for i in range(1, 4):
        frames_dir = f'../data/frames/0{i}'
        features_dir = f'../data/features/0{i}'
        os.makedirs(features_dir, exist_ok=True)

        # List all frames
        frames = os.listdir(frames_dir)

        # Use tqdm to display progress bar
        for frame in tqdm(frames, desc=f'Extracting features for video 0{i}', unit='frame'):
            frame_path = os.path.join(frames_dir, frame)
            features = extract_features(frame_path)
            features_path = os.path.join(features_dir, frame.replace('.jpg', '.npy'))
            np.save(features_path, features)

    for i in range(1, 10, 2):
        frames_dir = f'../data/frames_test/0{i}'
        features_dir = f'../data/features_test/0{i}'
        os.makedirs(features_dir, exist_ok=True)

        # List all frames
        frames = os.listdir(frames_dir)

        # Use tqdm to display progress bar
        for frame in tqdm(frames, desc=f'Extracting features for video 0{i}', unit='frame'):
            frame_path = os.path.join(frames_dir, frame)
            features = extract_features(frame_path)
            features_path = os.path.join(features_dir, frame.replace('.jpg', '.npy'))
            np.save(features_path, features)
