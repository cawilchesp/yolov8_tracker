from pathlib import Path

# Folder parameters
ROOT = Path('D:/Data')
SOURCE_FOLDER = ROOT
OUTPUT_FOLDER = ROOT
MODEL_FOLDER = ROOT / 'models' / 'yolov8'

# Source parameters
INPUT_VIDEO = 'paradero.mp4'
OUTPUT_NAME = 'paradero_yolov8'

# Deep Learning model configuration
MODEL_WEIGHTS = 'yolov8x.pt'

# Inference configuration
CLASS_FILTER = [0,1,2,3,5,7]
IMAGE_SIZE = 640
CONFIDENCE = 0.25
NMS_IOU = 0.7
