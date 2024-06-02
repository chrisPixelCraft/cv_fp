import json
import torch
from torch.nn.utils.rnn import pad_sequence

def filename2id(x):
    # print(f"[filename2id] x: {x}")
    return int(x.split('_')[1].split('.')[0])

def dir_filename2id(x) -> tuple:
    return (int(x.split('/')[0]), int(x.split('_')[1].split('.')[0]))

def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels

def pad_collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels = torch.tensor(labels)
    return features_padded.unsqueeze(1), labels