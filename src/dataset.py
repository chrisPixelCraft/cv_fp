import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import json
import numpy as np
from model import CNNRNNModel
from utils import filename2id, load_labels
            
class DoorStateDatasetTrain(Dataset):
    def __init__(self, features_dir, labels, label_idx, num_of_frames=1, spacing=5):
        super().__init__()
        if num_of_frames not in [1, 3]:
            raise ValueError(f"Invalid num_of_frames in [1, 5, 10]: {num_of_frames}")
        
        self.data = []
        self.features_dir = features_dir
        self.num_of_frames = num_of_frames
        self.spacing = spacing
        self.labels = labels
        self.label_idx = label_idx
        self.total_frames = len(self.labels)
        
        for idx in range(self.total_frames):
            frame_path = self.labels[idx]['frames']
            label = self.labels[idx]['label']
            feature_path = os.path.join(self.features_dir, frame_path[:-4] + ".npy")
            features = np.load(feature_path)
            features = torch.tensor(features, dtype=torch.float32)
            self.data.append((features, label))
        if self.num_of_frames > 1:
            self.data = self.spacing * [self.data[0]] + self.data + self.spacing * [self.data[-1]]

    def __len__(self):
        return len(self.label_idx)

    def __getitem__(self, idx):
        center_idx = self.label_idx[idx]
        if self.num_of_frames == 1:
            return self.data[self.label_idx[idx]]
        else:
            center_idx = self.label_idx[idx]
            label = self.data[center_idx][1]
            f1 = self.data[center_idx - self.spacing][0]
            f2 = self.data[center_idx][0]
            f3 = self.data[center_idx + self.spacing][0]
            ret = torch.cat([f1, f2, f3])
            # ret = torch.cat(
            #     [d[0] for d in self.data[center_idx - self.spacing: center_idx + self.spacing + 1: self.spacing]]
            # )
            
            # print(idx, ret.shape)
            # print('hi')
            return ret, label
        
class DoorStateDatasetTest(Dataset):
    def __init__(self, features_dir, num_of_frames=1, spacing=5):
        self.features_dir = features_dir
        self.feature_files = sorted(os.listdir(self.features_dir), key=lambda x: filename2id(x))
        self.num_of_frames = num_of_frames
        self.spacing = spacing
        self.data = []
        
        for feature_file in self.feature_files:
            feature_path = os.path.join(self.features_dir, feature_file)

            # feature_path = np.array(feature_path)
            features = np.load(feature_path)
            features = torch.tensor(features, dtype=torch.float32)
            # Add an extra dimension for sequence length
            # features = features.unsqueeze(0)  # Shape: [1, input_size]
            self.data.append(features)
        if self.num_of_frames > 1:
            self.data = self.spacing * [self.data[0]] + self.data + self.spacing * [self.data[-1]]


    def __len__(self):
        # Return the number of feature files
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.num_of_frames == 1:
            return self.data[idx]
        else:
            center_idx = idx
            f1 = self.data[center_idx - self.spacing]
            f2 = self.data[center_idx]
            f3 = self.data[center_idx + self.spacing]
            ret = torch.cat([f1, f2, f3]).unsqueeze(0)
            return ret
        # feature_file = self.feature_files[idx]
        # # feature_path = [f for f in os.listdir(self.features_dir)]
        # feature_path = os.path.join(self.features_dir, feature_file)
        # # print(f"Loading feature from: {feature_path}")  # Print the feature path

        # # feature_path = np.array(feature_path)
        # features = np.load(feature_path)
        # features = torch.tensor(features, dtype=torch.float32)
        # # Add an extra dimension for sequence length
        # features = features.unsqueeze(0)  # Shape: [1, input_size]

