from pathlib import Path

# Folder parameters
ROOT = Path('D:/Data')
SOURCE_FOLDER = ROOT
OUTPUT_FOLDER = ROOT

# Source parameters

INPUT_VIDEO = 'CHICO1_1_new.mp4'
OUTPUT_NAME = 'CHICO1_1_new_yolov10'

# Deep Learning model configuration
MODEL_ROOT = Path('D:/Data/models')
MODEL_FOLDER = MODEL_ROOT / 'yolo11'
MODEL_WEIGHTS = 'yolo11m.pt'

# Inference configuration
CLASS_FILTER = [0,1,2,3,5,7]
IMAGE_SIZE = 640
CONFIDENCE = 0.5
