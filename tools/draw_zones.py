import supervision as sv

import os
import cv2
import json
import argparse
import itertools
import numpy as np
from typing import Any, Optional, Tuple

import config
from tools.general import get_stream_frames_generator
import tools.messages as messages

# For debugging
from icecream import ic


KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

THICKNESS = 1
COLORS = sv.ColorPalette.DEFAULT
WINDOW_NAME = "Draw Zones"
POLYGONS = [[]]

current_mouse_position: Optional[Tuple[int, int]] = None


def resolve_source(source_path: str) -> Optional[np.ndarray]:
    # if not os.path.exists(source_path):
    #     return None

    # image = cv2.imread(source_path)
    # if image is not None:
    #     return image
    
    video_source = eval(source_path) if source_path.isnumeric() else source_path
    if video_source is not None:
        cap = cv2.VideoCapture(video_source)
        success, frame = cap.read()
        if not success:
            print('Error Camera Connection')
            return None
        cap.release()
        return frame


def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:
        current_mouse_position = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
        POLYGONS[-1].append((x, y))


def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()
    for idx, polygon in enumerate(POLYGONS):
        color = (
            COLORS.by_idx(idx).as_bgr()
            if idx < len(POLYGONS) - 1
            else sv.Color.WHITE.as_bgr()
        )

        if len(polygon) > 1:
            for i in range(1, len(polygon)):
                cv2.line(
                    img=image,
                    pt1=polygon[i - 1],
                    pt2=polygon[i],
                    color=color,
                    thickness=THICKNESS,
                )
            if idx < len(POLYGONS) - 1:
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=current_mouse_position,
                color=color,
                thickness=THICKNESS,
            )
    cv2.imshow(WINDOW_NAME, image)


def close_and_finalize_polygon(image: np.ndarray, original_image: np.ndarray) -> None:
    if len(POLYGONS[-1]) > 2:
        cv2.line(
            img=image,
            pt1=POLYGONS[-1][-1],
            pt2=POLYGONS[-1][0],
            color=COLORS.by_idx(0).as_bgr(),
            thickness=THICKNESS,
        )
    POLYGONS.append([])
    image[:] = original_image.copy()
    redraw_polygons(image)
    cv2.imshow(WINDOW_NAME, image)


def redraw_polygons(image: np.ndarray) -> None:
    for idx, polygon in enumerate(POLYGONS[:-1]):
        if len(polygon) > 1:
            color = COLORS.by_idx(idx).as_bgr()
            for i in range(len(polygon) - 1):
                cv2.line(
                    img=image,
                    pt1=polygon[i],
                    pt2=polygon[i + 1],
                    color=color,
                    thickness=THICKNESS,
                )
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=polygon[0],
                color=color,
                thickness=THICKNESS,
            )


def save_polygons_to_json(polygons, target_path):
    data_to_save = polygons if polygons[-1] else polygons[:-1]
    with open(target_path, "w") as f:
        json.dump(data_to_save, f)


def main(
    source: str,
    calibration: str
) -> None:
    global current_mouse_position
    original_image = resolve_source(source_path=source)
    if original_image is None:
        messages.step_message(step_count, 'No se pudo cargar la imagen ❌')
        return

    image = original_image.copy()
    cv2.imshow(WINDOW_NAME, image)
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_ENTER or key == KEY_NEWLINE:
            close_and_finalize_polygon(image, original_image)
            messages.step_message(next(step_count), 'Polígono terminado ✅')
        elif key == KEY_ESCAPE:
            POLYGONS[-1] = []
            current_mouse_position = None
            messages.step_message(step_count, 'Polígono cancelado ❌')
        elif key == KEY_SAVE:
            save_polygons_to_json(POLYGONS, calibration)
            messages.step_message(next(step_count), f"Polígonos guardados en {calibration} ✅")
            break
        redraw(image, original_image)
        if key == KEY_QUIT:
            messages.step_message('Error', 'Proceso cancelado ❌')
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    step_count = itertools.count(1)
    main(
        source=f"{config.SOURCE_FOLDER}/{config.INPUT_VIDEO}",
        calibration=f"{config.SOURCE_FOLDER}/{config.JSON_NAME}"
    )