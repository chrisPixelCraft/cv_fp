import os
import json
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNNRNNModel
from PIL import Image
from tqdm import tqdm
# from train import DoorStateDataset
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

from utils import filename2id
from dataset import DoorStateDatasetTest

# config
# data
frames_per_input = 35
spacing = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    ret = np.convolve(interval, window, 'same')
    return ret

def median_filter(data, kernel_size):
    k = kernel_size // 2
    # 边缘填充，确保输出大小与输入相同
    padded_data = np.pad(data, (k, k), mode='reflect')
    # 存储过滤结果
    filtered_data = np.zeros_like(data)
    
    # 对每个数据点应用中值滤波
    for i in range(len(data)):
        # 提取当前窗口
        window = padded_data[i:i + kernel_size]
        # 计算窗口的中位数
        filtered_data[i] = np.median(window)
    
    return filtered_data

def load_model(model_path, input_size, rnn_hidden_size, num_classes, rnn_layers):
    model = CNNRNNModel(input_size=input_size, rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, rnn_layers=rnn_layers, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.to(device)
    model.eval()
    return model


def guess_door_states(model, test_loader):
    """ Use the model to predict the frames for door states. """
    # print(frame_dir)
    # frames = [f for f in os.listdir(frame_dir)]
    # print(frames)

    all_outputs = []

    with torch.no_grad():
        for fet_batch, flow_batch in test_loader:
            fet_batch = fet_batch.to(next(model.parameters()).device)  # Ensure the batch is on the same device as the model
            # print(batch.shape)
            flow_batch = flow_batch.to(next(model.parameters()).device)
            outputs = model(fet_batch, flow_batch)
            all_outputs.append(outputs.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)
    return all_outputs



def scan_videos(frame_dir, feature_dir, model, model_name='CNNRNN'):
    """Scan the specified directory for feature_test folders and generate JSON annotations."""
    frame_dir_full = frame_dir
    frame_dir = [f for f in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, f))]

    print(frame_dir_full)

    videos_info = []
    batch_size = 16

    for frame in tqdm(frame_dir, desc="Processing videos"):
        # feature_dir = os.path.join(feature_dir, video_dir)
        video_dir = os.path.join(frame, '.mp4')
        feature_dir_in_frame = os.path.join(feature_dir, frame)
        frame_subdir  = os.path.join(frame_dir_full, frame)
        test_dataset = DoorStateDatasetTest(feature_dir_in_frame, num_of_frames=frames_per_input, spacing=spacing)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle= False)

        # opening_frame, closing_frame = guess_door_states(model, frame_dir_full , test_loader)

        outputs = guess_door_states(model, test_loader)
        predicted_logits = outputs.squeeze()
        # predict_max = np.max(predicted_logits, axis=0)
        # predict_min = np.min(predicted_logits, axis=0)
        # predicted_logits = (predicted_logits-predict_min) / (predict_max-predict_min)

        # print(predicted_logits.shape)
        # print(predict_max.shape)
        for i in range(3):
            predicted_logits[:, i] = moving_average(predicted_logits[:, i], 10)

        # predicted_logits = median_filter(predicted_logits[:,])


        plot_predictions(frame, predicted_logits, model=model_name)

        frames = [f for f in os.listdir(frame_subdir)]
        # print(frames)

        opening_logits = predicted_logits[:, 1]  # Logits for "opening" class
        closing_logits = predicted_logits[:, 2]  # Logits for "closing" class
        # opening_frame = frames[np.argmax(opening_logits)] if np.max(opening_logits) > 0.5 else -1
        # closing_frame = frames[np.argmax(closing_logits)] if np.max(closing_logits) > 0.5 else -1
        opening_frame = []
        closing_frame = []

        for id, f in enumerate(frames[:-1]):
            c = predicted_logits[id, 0]
            cn = predicted_logits[id+1, 0]
            o = predicted_logits[id, 1]
            on = predicted_logits[id+1, 1]
            print(c, cn, o, on)
            threshold = 0
            if c < o and cn > on:
                closing_frame.append(id)
            elif c > o and cn < on:
                opening_frame.append(id) 

        videos_info.append({
            "video_filename": video_dir,
            "annotations": [
                {
                    "object": "Door",
                    "states": [
                        {
                            "state_id": 1,
                            "description": "Opening",
                            "guessed_frame": opening_frame  # Predicted opening frame
                        },
                        {
                            "state_id": 2,
                            "description": "Closing",
                            "guessed_frame": closing_frame  # Predicted closing frame
                        }
                    ]
                }
            ]
        })

    return videos_info

def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, 'w') as file:
        json.dump({"videos": videos_info}, file, indent=4)

def plot_predictions(frame, predicted_logits, model):
    os.makedirs("plots", exist_ok=True)
    # Plot the prediction logits
    print(f"Predicted logits for {frame}:")
    print(predicted_logits.shape)
    plt.plot(predicted_logits)
    plt.xlabel('Frame Index')
    plt.ylabel('Logits')
    plt.legend(['Closed', 'Opened', 'Changing'])
    plt.title(f"Prediction for {frame}({model})")
    plt.savefig(f"plots/prediction_{frame}_{model}.png")
    plt.clf()

def main():
    frame_dir = "../data/frames_test"  # Specify the directory containing test video frames
    feature_dir = "../data/features_test_101" # Specify the directory containing test features by processing test video frames
    output_filename = "output.json"  # Output JSON file name
    model_path = "./models/model_small_22.pth"  # Path to the trained model file
    input_size = 2048  # Example input size, should match your precomputed feature size
    rnn_hidden_size = 512
    rnn_layers = 2  # Ensure this matches the training configuration
    num_classes = 3

    # Load the trained model
    model = load_model(model_path, input_size, rnn_hidden_size, num_classes, rnn_layers)

    # Process the videos and generate JSON annotations
    videos_info = scan_videos(frame_dir, feature_dir, model, model_path.split('/')[-1].split('.')[0])
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")

if __name__ == "__main__":
    main()


