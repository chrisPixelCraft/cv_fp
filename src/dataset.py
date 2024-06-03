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
        if num_of_frames % 2 == 0 or num_of_frames < 1:
            raise ValueError(f"Invalid num_of_frames: {num_of_frames}")
        
        self.data = []
        self.features_dir = features_dir
        self.num_of_frames = num_of_frames
        self.spacing = spacing
        self.farthest_frame = int((num_of_frames - 1) * spacing / 2)
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
            self.data = self.farthest_frame * [self.data[0]] + self.data + self.farthest_frame * [self.data[-1]]

    def __len__(self):
        return len(self.label_idx)

    def __getitem__(self, idx):
        center_idx = self.label_idx[idx]
        if self.num_of_frames == 1:
            return self.data[self.label_idx[idx]]
        else:
            label = self.data[center_idx][1]
            start_frame = center_idx - self.farthest_frame
            end_frame = center_idx + self.farthest_frame
            f = []
            for i in range(start_frame, end_frame+1, self.spacing):
                f.append(self.data[i][0])
            ret = torch.stack(f)
            
            return ret, label
        
class DoorStateDatasetTest(Dataset):
    def __init__(self, features_dir, num_of_frames=1, spacing=5):
        if num_of_frames % 2 == 0 or num_of_frames < 1:
            raise ValueError(f"Invalid num_of_frames: {num_of_frames}")
        
        self.features_dir = features_dir
        self.feature_files = sorted(os.listdir(self.features_dir), key=lambda x: filename2id(x))
        self.num_of_frames = num_of_frames
        self.spacing = spacing
        self.farthest_frame = int((num_of_frames - 1) * spacing / 2)
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
            self.data = self.farthest_frame * [self.data[0]] + self.data + self.farthest_frame * [self.data[-1]]


    def __len__(self):
        # Return the number of feature files
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.num_of_frames == 1:
            return self.data[idx]
        else:
            center_idx = idx
            start_frame = center_idx - self.farthest_frame
            end_frame = center_idx + self.farthest_frame
            f = []
            for i in range(start_frame, end_frame+1, self.spacing):
                f.append(self.data[i])
            ret = torch.stack(f)

            # print(ret.shape)
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

