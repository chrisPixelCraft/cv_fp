from feature_extraction import FeatureExtractor
from model import CNNRNNModel
from data_preparation import extract_frames
from torchvision import transforms
import torch
from PIL import Image
import os

def predict_door_state(model, video_path, output_dir, transform):
    extract_frames(video_path, output_dir)
    frame_paths = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir)])
    frames = [Image.open(frame_path).convert('RGB') for frame_path in frame_paths]
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        output = model(frames)
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == "__main__":
    video_path = '../data/raw_videos/new_video.mp4'
    output_dir = '../data/frames/new_video'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cnn = FeatureExtractor()
    model = CNNRNNModel(cnn, rnn_hidden_size=512, num_classes=4)
    model.load_state_dict(torch.load('path/to/saved_model.pth'))

    predicted_state = predict_door_state(model, video_path, output_dir, transform)
    print(f'Predicted Door State: {predicted_state}')


