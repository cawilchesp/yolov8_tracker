import supervision as sv

import cv2
import numpy as np

from icecream import ic


class SpeedAnalysis:
    def __init__(
        self,
        zone: np.array,
        width: int,
        height: int,
        fps: float,
        resolution_wh
    ) -> None:
        zone_source=zone.astype(np.float32)
        zone_target=np.array(
            [
                [0, 0], 
                [width - 1, 0], 
                [width - 1, height - 1], 
                [0, height - 1]
            ]
        ).astype(np.float32)
        self.fps = fps

        self.m = cv2.getPerspectiveTransform(zone_source, zone_target)

        self.line_thickness = int(sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh) * 0.5)
        self.text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh) * 0.5


    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        
        return transformed_points.reshape(-1, 2)
    

    def speed_calculation(self, new_position: tuple, last_position: tuple) -> tuple:
        t1, cx1, cy1, _, _, s1 = last_position
        t2, cx2, cy2 = new_position

        trx1, try1 = self.transform_points(points=np.array([cx1, cy1])).astype(int).tolist()[0]
        trx2, try2 = self.transform_points(points=np.array([cx2, cy2])).astype(int).tolist()[0]

        distance = abs(np.sqrt((try2-try1)**2 + (trx2-trx1)**2)) / 100
        time_diff = (t2 - t1) / self.fps

        speed = distance / time_diff * 3.6
        
        return speed
    
    def speed_estimation(self, object_data):
        speed_list = [data[-1] for data in object_data]
        speed_average = sum(speed_list) / len(speed_list)

        return speed_average


    def annotate(self, speed_track: dict, scene: np.array) -> np.array:
        for object_id, object_data in speed_track.items():
                frame_number, (x1, y1, x2, y2), speed_average = object_data

                point_X = x1 + (x2 - x1) / 2
                point_y = y1 + 20

                scene = sv.draw_text(
                    scene=scene,
                    text=f"{speed_average:.0f} km/h",
                    text_anchor=sv.Point(point_X, point_y),
                    background_color=sv.Color.RED,
                    text_color=sv.Color.WHITE,
                    text_scale=self.text_scale,
                    text_thickness=self.line_thickness,
                    text_padding=1
                )

        return scene


    
    """
    from collections import defaultdict, deque


    ZONE_ANALYSIS = np.array([[530,128], [800,128], [1255,673], [0,673]])
    TARGET_WIDTH = 25
    TARGET_HEIGHT = 75
    TARGET = np.array( [ [0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1] ] )
    
    view_transformer = ViewTransformer(source=ZONE_ANALYSIS, target=TARGET)
    coordinates = defaultdict(lambda: deque(maxlen=int(source_info.fps)))

    Example:

    points = view_transformer.transform_points(points=points).astype(int)
    
    object_labels =[]
    for tracker_id, [x, y] in zip(tracks.tracker_id, points):
        coordinates[tracker_id].append([frame_number, x, y])
        if len(coordinates[tracker_id]) < source_info.fps / 2:
            object_labels.append(f"#{tracker_id}")
        else:
            t_0, x_0, y_0 = coordinates[tracker_id][0]
            t_1, x_1, y_1 = coordinates[tracker_id][-1]

            distance = abs(np.sqrt((y_1-y_0)**2 + (x_1-x_0)**2))
            time_diff = (t_1 - t_0) / source_info.fps

            speed = distance / time_diff * 3.6
            object_labels.append(f"#{tracker_id} {int(speed)} Km/h")
    
    """