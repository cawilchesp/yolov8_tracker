from ultralytics import YOLOWorld
import supervision as sv

import torch
import cv2
from pathlib import Path
import datetime
import itertools

from imutils.video import FileVideoStream, WebcamVideoStream

from modules.model_loader import ModelSink
from modules.annotation import AnnotationSink

import config
from tools.video_info import VideoInfo
from tools.messages import source_message, progress_message, step_message
from tools.write_data import csv_append, write_csv


# For debugging
from icecream import ic


def main(
    source: str = '0',
    output: str = 'output',
    weights: str = 'yolov8m.pt',
    class_filter: list[int] = None,
    image_size: int = 640,
    confidence: int = 0.5,
) -> None:
    model = YOLOWorld(weights)

    model.set_classes(['car','bicycle','motorcycle','person'])

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(source)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    while cap.isOpened():
        ret, img = cap.read()

        if not ret: break

        results = model.predict(
            source=img,
            imgsz=image_size,
            conf=confidence,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=False,
        )

        detections = sv.Detections.from_ultralytics(results[0])

        annotated_frame = img.copy()

        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        out.write(annotated_frame)


        cv2.imshow("Output", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n")
            break



if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.SOURCE_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.OUTPUT_FOLDER}/{config.OUTPUT_NAME}",
        weights=f"{config.MODEL_FOLDER}/{config.MODEL_WEIGHTS}",
        # class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
    )