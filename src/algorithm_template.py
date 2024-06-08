from collections import deque
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # data
frames_per_input = 77
spacing = 1

def load_model(model_path, input_size, rnn_hidden_size, num_classes, rnn_layers):
    model = CNNRNNModel(input_size=input_size, rnn_hidden_size=rnn_hidden_size, num_classes=num_classes, rnn_layers=rnn_layers, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.to(device)
    model.eval()
    return model

class changing_noise_filter():
    def __init__(self, predictions, intersections) -> None:
        self.intersections = sorted(intersections)
        self.threshold_1 = 15
        self.window_2 = 25
        self.predictions = predictions
        print("intersections", self.intersections)

    def run(self):
        self.stage1()
        self.stage2()
        return self.fixframe()
        
    def stage1(self):
        # delete adjacent frames
        stage1_result = self.intersections
        for _ in range(len(self.intersections)):
            gap = {}
            for i in range(len(stage1_result)-1):
                gap[i] = stage1_result[i+1] - stage1_result[i]
            
            sorted_idx = sorted(range(len(gap)), key=lambda x: gap[x])
            deleted_frame = set()
            for idx in sorted_idx:
                if gap[idx] < self.threshold_1 and (idx not in deleted_frame) and (idx+1 not in deleted_frame):
                    deleted_frame.add(idx)
                    deleted_frame.add(idx+1)
            stage1_result = [stage1_result[i] for i in range(len(stage1_result)) if i not in deleted_frame]
            if len(deleted_frame) == 0:
                break

        self.stage1_result = stage1_result

    def stage2(self):
        # delete frames in window
        stage2_result = set()
        for i in range(len(self.stage1_result)):
            if i == 0:
                stage2_result.add(self.stage1_result[i])
            else:
                if self.stage1_result[i] - self.stage1_result[i-1] > self.window_2:
                    stage2_result.add(self.stage1_result[i])
                    stage2_result.add(self.stage1_result[i-1])
                else: 
                    d1 = self.find_distance(self.stage1_result[i-1])
                    d2 = self.find_distance(self.stage1_result[i])
                    if d1 > d2:
                        stage2_result.add(self.stage1_result[i-1])
                        stage2_result.discard(self.stage1_result[i])
                    else:
                        stage2_result.add(self.stage1_result[i])
                        stage2_result.discard(self.stage1_result[i-1])
        self.stage2_result = sorted(list(stage2_result))

    # def find_distance(self, frame):
    
    def find_distance(self, center):
        # center is the index of the center frame
        farthest = 40
        left_high = float('-inf')
        right_high = float('-inf')
        left_low = float('inf')
        right_low = float('inf')

        
        for i in range (2): 
            margin_left = max(0, center - farthest) 
            margin_right = min(center + farthest, len(self.predictions) - 1)
            # min_left = np.min(self.predictions[margin_left:center])
            min_right = np.min(self.predictions[center:margin_right, i]) 
            max_left = np.max(self.predictions[margin_left:center, i]) 
            min_left = np.min(self.predictions[margin_left:center, i]) 
            max_right = np.max(self.predictions[center:margin_right, i]) 
            center_val = self.predictions[center,i] 

            for j in range (1, farthest):
                left_j_margin = max(0, center-j)
                right_j_margin = min(center+j, len(self.predictions) - 1)

                # case 1  left high right low for opened, right high left low for closed 
                if max_left > center_val and min_right < center_val and i==0 : # i == 0 for opened 
                    if self.predictions[left_j_margin, i] > left_high:
                        left_high = self.predictions[left_j_margin, i]
                    if self.predictions[right_j_margin, i] < right_low:
                        right_low = self.predictions[right_j_margin, i]

                if max_right > center_val and min_left < center_val and i==1:
                    if self.predictions[right_j_margin, i] > right_high:
                        right_high = self.predictions[right_j_margin, i]
                    if self.predictions[left_j_margin, i] < left_low:
                        left_low = self.predictions[left_j_margin, i] 

                # case 2 left low right high for opened, right low left low for closed
                if min_left < center_val and max_right > center_val and i==0:
                    if self.predictions[left_j_margin, i] < left_low:
                        left_low = self.predictions[left_j_margin, i]
                    if self.predictions[right_j_margin, i] > right_high:
                        right_high = self.predictions[right_j_margin, i]
                
                if min_right < center_val and max_left > center_val and i==1:
                    if self.predictions[right_j_margin, i] < right_low:
                        right_low = self.predictions[right_j_margin, i]
                    if self.predictions[left_j_margin, i] > left_high:
                        left_high = self.predictions[left_j_margin, i]
        
        distance = ((left_high - right_low) + (right_high - left_low) )/2

        return distance
                
    def fixframe(self):
        fixframe = self.stage2_result[:]
        print(fixframe)
        state =  [True for i in range(len(fixframe))] # True for opening 

        for id in range(len(fixframe)):
            # print(id)
            # print(fixframe[id])
            c = self.predictions[fixframe[id], 0]
            cn = self.predictions[fixframe[id]+1, 0]
            o = self.predictions[fixframe[id], 1]
            on = self.predictions[fixframe[id]+1, 1]
            # print(c, cn, o, on)
            threshold = 0
            if c < o and cn > on:
                # closing_frame.append(id)
                # print('hello closing')
                state[id] = False
            elif c > o and cn < on:
                # opening_frame.append(id) 
                # print('hello opening')
                state[id] = True
        
        # print(state) 

        # TTF or FFT 
        start = 0
        end = 0
        deleted = set() # id in fixframe
        while end < len(fixframe) and end < len(fixframe):
            while state[end] == state[start]:
                end += 1
                if end == len(state):
                    break
            candidate = fixframe[start:end]
            print(candidate)
            distances = [self.find_distance(fixframe[start+i]) for i in range(len(candidate))]
            max_distance = max(distances)
            for i in range(len(candidate)):
                if distances[i] < max_distance:
                    deleted.add(start+i)
            start = end
            end = start + 1

        return [fixframe[i] for i in range(len(fixframe)) if i not in deleted]
                
        # for i in range(1, farthest): 
        #     if self.predictions[center-i] > left_high: 
        #         left_high = self.predictions[center-i] 
        #     else:
        #         break
            
        # if self.predictions[center-i] < left_low:
        #     left_low = self.predictions[center-i]
        
        


def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    ret = np.convolve(interval, window, 'same')
    return ret

def median_filter(data, kernel_size):
    k = kernel_size // 2
    # 边缘填充，确保输出大小与输入相同
    padded_data = np.pad(data, (k, k), mode='linear_ramp')
    # 存储过滤结果
    filtered_data = np.zeros_like(data)
    
    # 对每个数据点应用中值滤波
    for i in range(len(data)):
        # 提取当前窗口
        window = padded_data[i:i + kernel_size]
        # 计算窗口的中位数
        filtered_data[i] = np.median(window)
    
    return filtered_data


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
    batch_size = 32

    for frame in tqdm(frame_dir, desc="Processing videos"):
        # feature_dir = os.path.join(feature_dir, video_dir)
        video_dir = frame + ".mp4"
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
        f = (predicted_logits[:, 2] > predicted_logits[:, 1]) * (predicted_logits[:, 2] > predicted_logits[:, 0])
        possible_changing = np.array(range(predicted_logits.shape[0]))[f]
        # print(possible_changing)
        for i in range(3):
            predicted_logits[:, i] = median_filter(predicted_logits[:, i], 5)
            predicted_logits[:, i] = moving_average(predicted_logits[:, i], 9)
            pass

        plot_predictions(frame, predicted_logits, model=model_name)

        frames = [f for f in os.listdir(frame_subdir)]
        # print(frames)

        # opening_frame = frames[np.argmax(opening_logits)] if np.max(opening_logits) > 0.5 else -1
        # closing_frame = frames[np.argmax(closing_logits)] if np.max(closing_logits) > 0.5 else -1
        opening_frame = []
        closing_frame = []
        avg = np.mean(predicted_logits, axis=0)
        avg_open = avg[0]
        avg_close = avg[1]
        avg_changing = avg[2]

        # step 1. add possible point
        for id, f in enumerate(frames[:-1]):
            c = predicted_logits[id, 0]
            cn = predicted_logits[id+1, 0]
            o = predicted_logits[id, 1]
            on = predicted_logits[id+1, 1]
            # print(c, cn, o, on)
            threshold = 0
            if c < o and cn > on:
                closing_frame.append(id)
            elif c > o and cn < on:
                opening_frame.append(id) 

        # step 2. noise filter
        filter = changing_noise_filter(predicted_logits, closing_frame + opening_frame)
        possible_changing = filter.run()
        # print(type(possible_changing))
        # fixframe = changing_noise_filter(predicted_logits, possible_changing).fixframe()
        # print(possible_changing)


        # step 3. add possible point
        closing_frame = []
        opening_frame = []
        for id in possible_changing:
            c = predicted_logits[id, 0]
            cn = predicted_logits[id+1, 0]
            o = predicted_logits[id, 1]
            on = predicted_logits[id+1, 1]
            # print(c, cn, o, on)
            threshold = 0
            if c < o and cn > on:
                closing_frame.append(id)
            elif c > o and cn < on:
                opening_frame.append(id) 

        # step 4. delete unresonalbe point
        if len(closing_frame) > 0 and len(opening_frame) > 0:
            if closing_frame[0] < opening_frame[0]:
                closing_frame = closing_frame[1:]
            if closing_frame[-1] < opening_frame[-1]:
                opening_frame = opening_frame[:-1]

        print("opening_frame", opening_frame)
        print("closing_frame", closing_frame)

        json_state = []
        id = 1
        for op in opening_frame:
            json_state.append({
                "state_id": id,
                "description": "Opening",
                "guessed_frame": op-20  # Predicted opening frame
            })
            id += 2
        id = 2
        for cl in closing_frame:
            json_state.append({
                "state_id": id,
                "description": "Closing",
                "guessed_frame": cl-20  # Predicted closing frame
            })
            id += 2
        json_state.sort(key=lambda x: x["state_id"])
        

        videos_info.append({
            "video_filename": video_dir,
            "annotations": [
                {
                    "object": "Door",
                    "states": json_state
                }
            ]
        })

    return videos_info

def generate_json(output_filename, videos_info):
    """Generate a JSON file with the provided video information."""
    with open(output_filename, 'w') as file:
        json.dump({"videos": videos_info}, file, indent=4)

def plot_predictions(frame, predicted_logits, model):
    p = "./plots"
    os.makedirs(p, exist_ok=True)
    # Plot the prediction logits
    print(f"Predicted logits for {frame}:")
    print(predicted_logits.shape)
    # avg = np.mean(predicted_logits[:, 2])
    # avg = np.ones_like(predicted_logits[:, 2]) * avg
    # plt.plot(avg)
    plt.plot(predicted_logits)
    plt.xlabel('Frame Index')
    plt.ylabel('Logits')
    plt.legend(['Closed', 'Opened', 'Changing'])
    plt.title(f"Prediction for {frame}({model})")
    plt.savefig(f"{p}/aa{frame}.png")
    plt.clf()

def main():
    frame_dir = "../data/frames_test"  # Specify the directory containing test video frames
    feature_dir = "../data/features_test_101" # Specify the directory containing test features by processing test video frames
    output_filename = "output.json"  # Output JSON file name
    model_path = "./GRU2_51/models/model_epoch_49.pth"  # Path to the trained model file
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


