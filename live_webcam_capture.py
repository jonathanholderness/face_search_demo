# Requirements
# pip install ultralytics
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

import torch
import torchvision
from ultralytics import YOLO

# Check for CUDA device and set it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Avaliable device: {device}')


# Detection model - Live source
model = YOLO('yolov8n.pt').to('cuda')
print('Starting video')
results = model(source=0, show=True, save=False, stream=True)
for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

