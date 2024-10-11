import supervision as sv

import cv2
import torch
import datetime
import itertools
from tqdm import tqdm
from pathlib import Path

from modules.model_loader import ModelLoader
from modules.annotation import Annotation

import config
import tools.messages as messages
import tools.write_data as write_data
from tools.video_info import VideoInfo

# For debugging
from icecream import ic


def main(
    source: str = '0',
    output: str = 'output',
    weights: str = 'yolo11m.pt',
    class_filter: list[int] = None,
    image_size: int = 640,
    confidence: int = 0.25,
    samples: int = 100
) -> None:
    # Initialize video source
    source_info = VideoInfo(source=source)
    messages.step_message(next(step_count), 'Video Source Initialized ✅')
    messages.source_message(source_info)

    # Check GPU availability
    messages.step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize model
    yolo_tracker = ModelLoader(
        weights_path=weights,
        image_size=image_size,
        confidence=confidence,
        class_filter=class_filter )
    messages.step_message(next(step_count), f"{Path(weights).stem.upper()} Model Initialized ✅")

    # show_image size
    scaled_width = 1280 if source_info.width > 1280 else source_info.width
    scaled_height = int(scaled_width * source_info.height / source_info.width)
    scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    # Annotators
    annotator = Annotation(
        source_info=source_info,
    )

    # Start autolabelling process
    messages.step_message(next(step_count), 'Autolabelling Started ✅')

    # cap = cv2.VideoCapture(source)
    target = f"{output}/{Path(source).stem}"
    stride = round(source_info.total_frames / samples)
    frame_generator = sv.get_video_frames_generator(source_path=source, stride=stride)
    image_count = 0
    
    time_start = datetime.datetime.now()
    with sv.ImageSink(target_dir_path=target) as sink:
        for image in tqdm(frame_generator, total=round(source_info.total_frames / stride), unit='frames'):
            sink.save_image(image=image)
            txt_name = Path(sink.image_name_pattern.format(image_count)).stem

            annotated_image = image.copy()

            # Run YOLOv8 inference
            results = yolo_tracker.detect(image=image)
            
            # Saving detection results
            output_data = []
            output_data = write_data.txt_append(output_data, results)
            write_data.write_txt(f"{target}/{txt_name}.txt", output_data)

            image_count += 1
            
            detections = sv.Detections.from_ultralytics(results)
            
            # Draw annotations
            annotated_image = annotator.on_detections(detections=detections, scene=image)

            # View live results
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n")
                break

    # Print total time elapsed
    messages.step_message(next(step_count), f"Elapsed Time: {(datetime.datetime.now() - time_start).total_seconds():.2f} s")


if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.SOURCE_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.DATASET_FOLDER}/{config.OUTPUT_NAME}",
        weights=f"{config.MODEL_FOLDER}/{config.MODEL_WEIGHTS}",
        # class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        samples=config.SAMPLES
    )