import os
import json
import pickle
import torch
import numpy as np
from torchvision import models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNNRNNModel
from PIL import Image
from tqdm import tqdm
from scipy.stats import mode

# from train import DoorStateDataset
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


from utils import filename2id
from dataset import DoorStateDatasetTest

# config
# data
frames_per_input = 35
spacing = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, input_size, rnn_hidden_size, num_classes, rnn_layers):
    model = CNNRNNModel(
        input_size=input_size,
        rnn_hidden_size=rnn_hidden_size,
        num_classes=num_classes,
        rnn_layers=rnn_layers,
        dropout=0.5,
    )
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    )
    model.to(device)
    model.eval()
    return model


def guess_door_states(model, test_loader, save_path):
    """Use the model to predict the frames for door states."""
    all_outputs = []

    with torch.no_grad():
        for fet_batch, flow_batch in test_loader:
            fet_batch = fet_batch.to(
                next(model.parameters()).device
            )  # Ensure the batch is on the same device as the model
            flow_batch = flow_batch.to(next(model.parameters()).device)
            outputs = model(fet_batch, flow_batch)
            all_outputs.append(outputs.cpu().numpy())

    all_outputs = np.concatenate(all_outputs, axis=0)

    # Save the predicted labels to a pickle file
    with open(save_path, "wb") as f:
        pickle.dump(all_outputs, f)

    return all_outputs

def find_mode(arr):
    # Step 1: Create a frequency dictionary
    frequency = {}
    for num in arr:
        if num in frequency:
            frequency[num] += 1
        else:
            frequency[num] = 1

    # Step 2: Find the maximum frequency
    max_count = max(frequency.values())

    # Step 3: Identify the mode(s)
    modes = [key for key, value in frequency.items() if value == max_count]

    return modes[0]

import numpy as np

def exponential_normalize(data, feature_range=(-1, 1)):
    # Step 1: Shift the data to be positive
    shift = np.abs(np.min(data))
    shifted_data = data + shift

    # Step 2: Apply the exponential function
    exp_data = np.exp(shifted_data)

    # Step 3: Normalize the exponential data to the range -1 to 1
    min_exp = np.min(exp_data)
    max_exp = np.max(exp_data)
    range_min, range_max = feature_range
    normalized_data = (exp_data - min_exp) / (max_exp - min_exp) * (range_max - range_min) + range_min

    return normalized_data



def moving_mode(labels, window_size):
    rounded_labels = np.round(labels)
    # print(rounded_labels)
    # print(np.shape(rounded_labels))
    # print(len(rounded_labels))
    smoothed_labels = np.zeros((len(labels),3))

    for i in range(window_size//2, len(rounded_labels), window_size+1):
        for j in range(3):

            start_index = max(0, i - window_size // 2)
            end_index = min(len(rounded_labels), i + window_size // 2 + 1)
            window = rounded_labels[start_index:end_index,j]
            # print(np.shape(window))
            # print(window)
            # print(find_mode(window))

            mode = np.ones(window_size+1)*find_mode(window)
            if i + window_size //2 <= len(rounded_labels):
                smoothed_labels[start_index:end_index,j] = mode

    # print(smoothed_labels)
    # print(np.shape(smoothed_labels))


        # mode_result = mode(window)
        # print(mode_result)



    # for i in range(half_window, len(labels) - half_window):
    #     window = labels[i - half_window:i + half_window + 1]
    #     smoothed_labels[i] = mode(window, keepdims=True)[0][0]

    # return smoothed_labels

    # Normalize smoothed labels to range -1 to 1

    normalized_labels = exponential_normalize(smoothed_labels)
    print(np.shape(normalized_labels))

    return normalized_labels

def find_turning_points(normalized_labels, threshold=1):
    inner_threshold = 0.35
    turning_points = []
    for i in range(1, len(normalized_labels) - 1):
        # Check for a turning point by comparing the change in labels
        # print(normalized_labels[i,0], normalized_labels[i,1])
        if (np.abs(normalized_labels[i,0]- normalized_labels[i,1]) > threshold and (np.abs(normalized_labels[i-1,0]-normalized_labels[i,0]) > inner_threshold or np.abs(normalized_labels[i+1,0]-normalized_labels[i,0]) > inner_threshold) and (np.abs(normalized_labels[i-1,1]-normalized_labels[i,1]) > inner_threshold or np.abs(normalized_labels[i+1,1]-normalized_labels[i,1]) > inner_threshold)):
            turning_points.append(i)
    return turning_points


def find_most_probable_frames(smoothed_labels, turning_points, label_opening_closing):
    best_frame = 0
    target_label = 0

    print(turning_points)

    for i in range(len(turning_points)//4):
        print(i)
        if label_opening_closing ==1:
            target_label = (turning_points[4*i]+turning_points[4*i+1])//2
        elif label_opening_closing == 2:
            target_label = (turning_points[4*i+2]+turning_points[4*i+3])//2



    # for i in range(len(predicted_labels)):
    #     start = max(0, i - window_size // 2)
    #     end = min(len(predicted_labels), i + window_size // 2)
    #     window = predicted_labels[start:end]
    #     count = np.sum(window == target_label)
    #     if count > max_count:
    #         best_frame = i
    #         max_count = count

    return target_label


def scan_videos(frame_dir, feature_dir, model, model_name="CNNRNN"):
    """Scan the specified directory for feature_test folders and generate JSON annotations."""
    frame_dir_full = frame_dir
    frame_dir = [
        f for f in os.listdir(frame_dir) if os.path.isdir(os.path.join(frame_dir, f))
    ]

    print(frame_dir_full)

    videos_info = []
    batch_size = 16
    predicted_label_dir = "../data/predicted_labels"
    os.makedirs(predicted_label_dir, exist_ok=True)

    for frame in tqdm(frame_dir, desc="Processing videos"):
        # feature_dir = os.path.join(feature_dir, video_dir)
        video_dir = frame + ".mp4"
        print(frame)
        print(video_dir)
        feature_dir_in_frame = os.path.join(feature_dir, frame)
        frame_subdir = os.path.join(frame_dir_full, frame)
        test_dataset = DoorStateDatasetTest(
            feature_dir_in_frame, num_of_frames=frames_per_input, spacing=spacing
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # opening_frame, closing_frame = guess_door_states(model, frame_dir_full , test_loader)

        outputs = guess_door_states(
            model, test_loader, f"{predicted_label_dir}/predicted_labels_{frame}.pkl"
        )
        predicted_logits = outputs.squeeze()
        # predicted_labels = np.argmax(predicted_logits, axis=1)
        # Apply the moving mode filter to smooth the predicted labels
        window_size = 12  # Adjust the window size as needed
        normalized_smoothed_labels = moving_mode(predicted_logits, window_size)
        print(np.shape(normalized_smoothed_labels))
        turning_points = find_turning_points(normalized_smoothed_labels)

        opening_frame = find_most_probable_frames(normalized_smoothed_labels, turning_points, 1)
        closing_frame = find_most_probable_frames(normalized_smoothed_labels, turning_points, 2)

        plot_predictions(frame, normalized_smoothed_labels, model=model_name)

        # frames = [f for f in os.listdir(frame_subdir)]
        # # print(frames)

        # opening_logits = predicted_logits[:, 1]  # Logits for "opening" class
        # closing_logits = predicted_logits[:, 2]  # Logits for "closing" class
        # opening_frame = frames[np.argmax(opening_logits)] if np.max(opening_logits) > 0.5 else -1
        # closing_frame = frames[np.argmax(closing_logits)] if np.max(closing_logits) > 0.5 else -1

        videos_info.append(
            {
                "video_filename": video_dir,
                "annotations": [
                    {
                        "object": "Door",
                        "states": [
                            {
                                "state_id": 1,
                                "description": "Opening",
                                "guessed_frame": opening_frame,  # Predicted opening frame
                            },
                            {
                                "state_id": 2,
                                "description": "Closing",
                                "guessed_frame": closing_frame,  # Predicted closing frame
                            },
                        ],
                    }
                ],
            }
        )

    return videos_info


def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, "w") as file:
        json.dump({"videos": videos_info}, file, indent=4)


def plot_predictions(frame, predicted_logits, model):
    os.makedirs("plots", exist_ok=True)
    # Plot the prediction logits
    print(f"Predicted logits for {frame}:")
    print(predicted_logits.shape)
    print(predicted_logits[:,0])
    plt.plot(predicted_logits)
    plt.xlabel("Frame Index")
    plt.ylabel("Logits")
    plt.legend(["Opened", "Closed", "Changing"])
    plt.title(f"Prediction for {frame}({model})")
    plt.savefig(f"plots/prediction_{frame}_{model}.png")
    plt.clf()


def main():
    frame_dir = (
        "../data/frames_test"  # Specify the directory containing test video frames
    )
    feature_dir = "../data/features_test_101"  # Specify the directory containing test features by processing test video frames
    output_filename = "output.json"  # Output JSON file name
    model_path = "./models/model_small_58.pth"  # Path to the trained model file
    input_size = 2048  # Example input size, should match your precomputed feature size
    rnn_hidden_size = 512
    num_classes = 3
    rnn_layers = 2  # Ensure this matches the training configuration

    # Load the trained model
    model = load_model(model_path, input_size, rnn_hidden_size, num_classes, rnn_layers)

    # Process the videos and generate JSON annotations
    videos_info = scan_videos(
        frame_dir, feature_dir, model, model_path.split("/")[-1].split(".")[0]
    )
    generate_json(output_filename, videos_info)
    print(f"Generated JSON file '{output_filename}' with video annotations.")


if __name__ == "__main__":
    main()

