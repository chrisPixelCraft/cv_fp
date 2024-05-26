import cv2
import os
import json
import numpy as np
import re
from tqdm import tqdm

from utils import dir_filename2id

# For sorting the frames in labels array, then convert the array to labels.json file
def extract_folder_and_frame_number(filename):
    match = re.search(r'(\d+)/frame_(\d+)\.jpg', filename)
    if match:
        folder_number = int(match.group(1))
        frame_number = int(match.group(2))
        return folder_number, frame_number
    return None, None

def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    return frame_count

def generate_labels(ground_truth_path, output_labels_path, frame_count):
    # Load ground truth annotations
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)

    # Create labels list
    labels = []
    video_count = 0 # to trace the value in frame_count array

    # Process each video and its annotations
    for video in ground_truth["videos"]:
        video_filename = video["video_filename"]
        video_base = os.path.splitext(video_filename)[0]
        annotations = video["annotations"]

        for annotation in annotations:
            states = annotation["states"]
            opening_end_frame = 0
            closing_start_frame = 0

            for state in states:
                state_id = state["state_id"]
                description = state["description"]
                start_frame = state["start_frame"]
                end_frame = state.get("end_frame") or state.get("half_open_frame")

                frame_sequence = [f"{video_base}/frame_{i}.jpg" for i in range(start_frame, end_frame + 1)]
                for frame in frame_sequence:

                    labels.append({
                        "frames": frame,
                        "label": state_id  # Use state_id directly as the label
                    })

                if(state_id == 1):  # read the start_frame of opening state
                    frame_sequence_closed_before_opening = [f"{video_base}/frame_{i}.jpg" for i in range(0, start_frame)]
                    for frame in frame_sequence_closed_before_opening:
                        labels.append({
                            "frames": frame,
                            "label": 0  # Use 0 as the label for "closed"
                        })
                    opening_end_frame = end_frame

                if(state_id ==2): #read the end frame of closing state
                    frame_sequence_closed_after_closing = [f"{video_base}/frame_{i}.jpg" for i in range(end_frame+1, int(frame_count[video_count]))]
                    for frame in frame_sequence_closed_after_closing:
                        labels.append({
                            "frames": frame,
                            "label": 0  # Use 0 as the label for "closed"
                        })
                    # print(frame_sequence_closed_after_closing)
                    # print(frame_count)
                    closing_start_frame = start_frame

            frame_sequence_opened = [f"{video_base}/frame_{i}.jpg" for i in range(opening_end_frame+1, closing_start_frame)]
            for frame in frame_sequence_opened:
                labels.append({
                    "frames": frame,
                    "label": 3  # Use 3 as the label for "opened"
                })

        video_count += 1

    # Sort the labels by the folder number and frame number of the first frame in each frame sequence
    labels.sort(key=lambda x: dir_filename2id(x["frames"]))
    # print(labels)
    # print(np.shape(labels))

    # Save labels to labels.json
    os.makedirs(os.path.dirname(output_labels_path), exist_ok=True)
    with open(output_labels_path, 'w') as f:
        json.dump(labels, f, indent=4)

    print(f"Labels saved to {output_labels_path}")


if __name__ == "__main__":
    frame_count = np.zeros((3,1))
    frame_count_test = np.zeros((10,1))

    # prepare frames for training videos
    for i in tqdm(range(1, 4), desc="Processing videos"):
        video_path = f"../data/raw_videos/0{i}.mp4"
        output_dir = f"../data/frames/0{i}"
        os.makedirs(output_dir, exist_ok=True)
        frame_count[i-1] = extract_frames(video_path, output_dir)

    # prepare frames for testing videos
    for i in tqdm(range(1, 10, 2), desc="Processing videos"):
        video_path = f"../data/test_videos/0{i}.mp4"
        output_dir = f"../data/frames_test/0{i}"
        os.makedirs(output_dir, exist_ok=True)
        frame_count_test[i-1] = extract_frames(video_path, output_dir)
        # print(frame_count_test)

    ground_truth_path = '../ground_truth_annotations.json'
    output_labels_path = '../data/labels/labels.json'
    generate_labels(ground_truth_path, output_labels_path,frame_count)


