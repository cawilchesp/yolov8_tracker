from ultralytics import YOLO
import supervision as sv

import torch
import numpy as np
from typing import List


class DetectionSink:
    def __init__(
        self,
        weights_path: str,
        image_size: int = 640,
        confidence: float = 0.5,
        class_filter: List[int] = None,
        nms_threshold: float = 0.7
    ) -> None:
        self.model = YOLO(weights_path)
        self.image_size = image_size        
        self.confidence = confidence
        self.class_filter = class_filter
        self.nms_threshold = nms_threshold

    def detect(self, image: np.array) -> sv.Detections:
        results = self.model(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence,
            classes=self.class_filter,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(results).with_nms(threshold=self.nms_threshold)
        
        return detections