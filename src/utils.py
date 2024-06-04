import json
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed):
    ''' set random seeds '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

def filename2id(x):
    # print(f"[filename2id] x: {x}")
    return int(x.split('_')[1].split('.')[0])

def dir_filename2id(x) -> tuple:
    return (int(x.split('/')[0]), int(x.split('_')[1].split('.')[0]))

def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels

def pad_collate_fn_unsqueeze(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels = torch.tensor(labels)
    return features_padded.unsqueeze(1), labels

def pad_collate_fn(batch):
    features, labels = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels = torch.tensor(labels)
    return features_padded, labels

def plot_learning_curve(num_epochs, curve, plot_save_path):
    legend = []
    for key, value in curve.items():
        plt.plot(range(1, num_epochs+1), value)
        legend.append(key)
        
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Loss vs. Epoch')
    plt.legend(legend)
    plt.savefig(plot_save_path)
    plt.close()