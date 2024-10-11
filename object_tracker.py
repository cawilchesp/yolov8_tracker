from ultralytics import YOLO
import supervision as sv

import cv2
import torch
import datetime
import itertools
import numpy as np
from pathlib import Path
from collections import defaultdict, deque
from shapely.geometry import Point, Polygon

from imutils.video import FileVideoStream, WebcamVideoStream

from modules.model_loader import ModelLoader
from modules.annotation import Annotation
from modules.zone_analysis import ZoneAnalysis
from modules.speed_analysis import SpeedAnalysis

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
    confidence: int = 0.5,
    calibration: str = None,
) -> None:
    # Initialize video source
    source_info = VideoInfo(source=source)
    messages.step_message(next(step_count), 'Video Source Initialized ✅')
    messages.source_message(source_info)

    # Check GPU availability
    messages.step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize YOLOv8 model
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
        trace=True,
        mask=True
    )

    # # Initialize zones and calibration
    # zone_data = ZoneAnalysis(json_path=calibration)

    # # Initialize speed estimation
    # speed_data = SpeedAnalysis(
    #     zone = zone_data.calibration_zone,
    #     width = zone_data.calibration_width,
    #     height = zone_data.calibration_height,
    #     fps=source_info.fps,
    #     resolution_wh=source_info.resolution_wh
    # )
    object_track = defaultdict(lambda: deque(maxlen=10))
    
    # Start video tracking processing
    messages.step_message(next(step_count), 'Video Tracking Started ✅')
    
    if source_info.source_type == 'stream':
        video_stream = WebcamVideoStream(src=eval(source) if source.isnumeric() else source)
        source_writer = cv2.VideoWriter(f"{output}_source.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))
    elif source_info.source_type == 'file':
        video_stream = FileVideoStream(source)
    output_writer = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))

    frame_number = 0
    output_data = []
    video_stream.start()
    time_start = datetime.datetime.now()
    fps_monitor = sv.FPSMonitor()
    # smoother = sv.DetectionsSmoother()
    try:
        while video_stream.more() if source_info.source_type == 'file' else True:
            fps_monitor.tick()
            fps_value = fps_monitor.fps

            image = video_stream.read()
            if image is None:
                print()
                break

            # YOLO inference
            results = yolo_tracker.track(image=image)
                
            # Save object data in list
            output_data = write_data.csv_append(output_data, frame_number, results)

            # Convert results to Supervision format
            detections = sv.Detections.from_ultralytics(results)

            # # Analysis
            # if detections.tracker_id is not None:
            #     speed_track = dict()
            #     position_track = dict()
            #     for tracker_id, (x1, y1, x2, y2) in zip(detections.tracker_id, detections.xyxy):
            #         cx = x1 + (x2 - x1) / 2
            #         cy = y1 + (y2 - y1) / 2

            #         if not tracker_id in object_track:
            #             object_track[tracker_id].append((frame_number, cx, cy, 0, 0, 0.0))
            #         else:
            #             (ox, oy) = zone_data.orientation(last_position=object_track[tracker_id][-1], new_position=(cx, cy))
            #             speed = speed_data.speed_calculation(last_position=object_track[tracker_id][-1], new_position=(frame_number, cx, cy))
            #             object_track[tracker_id].append((frame_number, cx, cy, ox, oy, speed))

            #             speed_average = speed_data.speed_estimation(object_track[tracker_id])
            #             speed_track[tracker_id] = (frame_number, (x1, y1, x2, y2), speed_average)

            #             directions = zone_data.invasion_estimation((cx, cy), (ox, oy))
            #             position_track[tracker_id] = (frame_number, (x1, y1, x2, y2), directions)
                  
            # # Draw zones
            # annotated_image = zone_data.annotate_zones(scene=image)

            # Suavizar las cajas de detección
            # if detections.tracker_id is not None:
            #     detections = smoother.update_with_detections(detections)
            
            # Draw annotations
            annotated_image = annotator.on_detections(detections=detections, scene=image)

            # Draw masks
            # annotated_image = annotation_sink.on_masks(detections=detections, scene=image)

            # # Draw speed labels
            # if detections.tracker_id is not None:
            #     annotated_image = speed_data.annotate(
            #         scene=annotated_image,
            #         speed_track=speed_track
            #     )

            # # Draw invasion alert
            # if detections.tracker_id is not None:
            #     annotated_image = zone_data.annonate_alert(
            #         scene=annotated_image,
            #         position_track=position_track
            #     )

            # Save results
            output_writer.write(annotated_image)
            if source_info.source_type == 'stream': source_writer.write(image)

            # Print progress
            messages.progress_message(frame_number, source_info.total_frames, fps_value)
            frame_number += 1

            # View live results
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n")
                break

    except KeyboardInterrupt:
        messages.step_message(next(step_count), 'End of Video ✅')
    messages.step_message(next(step_count), 'Saving Detections in CSV file ✅')
    write_data.write_csv(f"{output}.csv", output_data)
    
    messages.step_message(next(step_count), f"Elapsed Time: {(datetime.datetime.now() - time_start).total_seconds():.2f} s")
    output_writer.release()
    if source_info.source_type == 'stream': source_writer.release()
    
    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.SOURCE_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.OUTPUT_FOLDER}/{config.OUTPUT_NAME}",
        weights=f"{config.MODEL_FOLDER}/{config.MODEL_WEIGHTS}",
        # class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
        calibration=f"{config.SOURCE_FOLDER}/{config.JSON_NAME}"
    )
