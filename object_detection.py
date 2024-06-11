import supervision as sv

import torch
import cv2
from pathlib import Path
import datetime
import itertools

from imutils.video import FileVideoStream, WebcamVideoStream

from sinks.detection_sink import DetectionSink
from sinks.annotation_sink import AnnotationSink

import config
from tools.video_info import VideoInfo
from tools.messages import source_message, progress_message, step_message
from tools.write_csv import output_append, write_csv

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
    # Initialize video source
    source_info, source_flag = VideoInfo.get_source_info(source)
    step_message(next(step_count), 'Origen del Video Inicializado ✅')
    source_message(source, source_info)

    # Check GPU availability
    step_message(next(step_count), f"Processor: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

    # Initialize YOLOv10 model
    detection_sink = DetectionSink(
        weights_path=weights,
        image_size=image_size,
        confidence=confidence,
        class_filter=class_filter )
    step_message(next(step_count), f"{Path(weights).stem.upper()} Model Initialized ✅")

    # show_image size
    scaled_width = 1280 if source_info.width > 1280 else source_info.width
    scaled_height = int(scaled_width * source_info.height / source_info.width)
    scaled_height = scaled_height if source_info.height > scaled_height else source_info.height

    # Annotators
    annotation_sink = AnnotationSink(
        source_info=source_info
    )

    # Start video detection processing
    step_message(next(step_count), 'Video Detection Started ✅')
    
    if source_flag == 'stream':
        video_stream = WebcamVideoStream(src=eval(source) if source.isnumeric() else source)
        source_writer = cv2.VideoWriter(f"{output}_source.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))
    elif source_flag == 'video':
        video_stream = FileVideoStream(source)
    output_writer = cv2.VideoWriter(f"{output}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), source_info.fps, (source_info.width, source_info.height))

    frame_number = 0
    output_data = []
    video_stream.start()
    time_start = datetime.datetime.now()
    fps_monitor = sv.FPSMonitor()
    try:
        while video_stream.more() if source_flag == 'video' else True:
            fps_monitor.tick()
            fps_value = fps_monitor.fps

            image = video_stream.read()
            if image is None:
                print()
                break

            # YOLO inference
            detections = detection_sink.detect(image=image)
                
            # Save object data in list
            output_data = output_append(output_data, frame_number, detections)

            # Draw annotations
            annotated_image = annotation_sink.on_detections(detections=detections, image=image)

            # Save results
            output_writer.write(annotated_image)
            if source_flag == 'stream': source_writer.write(image)

            # Print progress
            progress_message(frame_number, source_info.total_frames, fps_value)
            frame_number += 1

            # View live results
            cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Output', int(scaled_width), int(scaled_height))
            cv2.imshow("Output", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n")
                break

    except KeyboardInterrupt:
        step_message(next(step_count), 'End of Video ✅')
    step_message(next(step_count), 'Saving Detections in CSV file ✅')
    write_csv(f"{output}.csv", output_data)
    
    step_message(next(step_count), f"Elapsed Time: {(datetime.datetime.now() - time_start).total_seconds():.2f} s")
    output_writer.release()
    if source_flag == 'stream': source_writer.release()
    
    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.SOURCE_FOLDER}/{config.INPUT_VIDEO}",
        output=f"{config.OUTPUT_FOLDER}/{config.OUTPUT_NAME}",
        weights=f"{config.MODEL_FOLDER}/{config.MODEL_WEIGHTS}",
        class_filter=config.CLASS_FILTER,
        image_size=config.IMAGE_SIZE,
        confidence=config.CONFIDENCE,
    )
