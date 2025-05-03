import torch
import numpy as np
from torch.utils.data import DataLoader
from model import MultiModalActionModel  # Import your trained model

# Assuming the model has been trained and saved previously
model = MultiModalActionModel(num_classes=4)  # Update num_classes as per your model
checkpoint = torch.load("best_model.pth")  # Load your saved model checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

def preprocess_and_infer(radar_data, video_data, radar_transform=None, video_transform=None, device='cuda'):
    # Preprocess radar and video data
    radar_tensor = torch.from_numpy(radar_data).float().to(device)  # Assuming radar_data is a numpy array
    video_tensor = torch.stack([video_transform(frame) for frame in video_data]).to(device)  # Apply transformation for each video frame

    # Inference with the model
    with torch.no_grad():
        radar_output, video_output = model(radar_tensor, video_tensor)  # Assuming the model outputs separate modalities
        final_output = radar_output + video_output  # You can change how to combine outputs based on your model's logic

    return final_output

def collect_and_infer_data(radar_queue, video_queue, radar_transform=None, video_transform=None, device='cuda'):
    try:
        while True:
            radar_data = radar_queue.get(timeout=0.1)  # Get radar data from queue
            video_frame = video_queue.get(timeout=0.1)  # Get video frame from queue

            # Assuming radar_data and video_frame are lists of frames; you can adjust based on your data format
            radar_data_buffer = [radar_data]
            video_data_buffer = [video_frame['frame']]

            # Perform inference on the collected data
            output = preprocess_and_infer(radar_data_buffer, video_data_buffer, radar_transform, video_transform, device)

            # Process or log the inference result
            print(f"Inference Result: {output}")

    except KeyboardInterrupt:
        print("[INFO] Stopping data collection and inference.")
