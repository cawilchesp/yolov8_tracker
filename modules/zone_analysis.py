import supervision as sv

import cv2
import json
import torch
import numpy as np
from typing import List
from shapely.geometry import Point, Polygon

# For debugging
from icecream import ic


class ZoneAnalysis:
    def __init__(
        self,
        json_path: str,
    ) -> None:
        self.json_path = json_path
        self.json_data = self.load_json(json_path)
        self.polygons = [np.array(polygon, np.int32) for polygon in self.json_data['zones']]

        self.calibration_zone = np.array(self.json_data['calibration']['zone'])
        self.calibration_width = self.json_data['calibration']['width']
        self.calibration_height = self.json_data['calibration']['height']
        self.zones = [
            sv.PolygonZone(
                polygon=polygon,
                triggering_anchors=(sv.Position.BOTTOM_CENTER,),
            )
            for polygon in self.polygons
        ]


    def load_json(self, file_path: str):
        with open(file_path, "r") as file:
            data = json.load(file)

            return data
        
    def orientation(self, new_position: tuple, last_position: tuple) -> tuple:
        _, cx1, cy1, _, _, _ = last_position
        cx2, cy2 = new_position

        orientation_x = cx2 - cx1
        orientation_y = cy2 - cy1

        return (orientation_x, orientation_y)
    
    def invasion_estimation(self, center: tuple[float, float], orientations: tuple[float, float]):
        cx, cy = center
        orientation_x, orientation_y = orientations

        inside_1 = Polygon(self.zones[0].polygon.tolist()).contains(Point(cx, cy))
        # inside_2 = Polygon(self.zones[1].polygon.tolist()).contains(Point(cx, cy))

        direction_x = 0
        if inside_1:
            direction_y = orientation_y * -1
        # elif inside_2:
        #     direction_y = orientation_y * 1
        else:
            direction_y = 0
        
        return (direction_x, direction_y)
    
    def annonate_alert(self, position_track: dict, scene: np.array) -> np.array:
        for object_position in position_track.values():
            _, (x1, y1, x2, y2), directions = object_position
            
            if directions[0] < 0 or directions[1] < 0:
                cv2.rectangle(
                    img=scene,
                    pt1=(int(x1), int(y1)),
                    pt2=(int(x2), int(y2)),
                    color=(0,0,255),
                    thickness=2,
                )

                scene = sv.draw_text(
                    scene=scene,
                    text="Invasion de Carril",
                    text_anchor=sv.Point(540, 50),
                    background_color=sv.Color.RED,
                    text_color=sv.Color.WHITE,
                    text_scale=2,
                    text_thickness=2
                )

        return scene
    
    def annotate_zones(self, scene: np.array) -> np.array:
        mask_image = scene.copy()
        COLORS = sv.ColorPalette.from_hex(["#FFFF55", "#3C76D1", "#FFFF55", "#FF5555", "#3CB44B"])
        
        for idx, zone in enumerate(self.zones):
            scene = cv2.fillPoly(
                scene,
                [zone.polygon],
                color=COLORS.by_idx(idx).as_bgr()
            )
        scene = cv2.addWeighted(
            scene, 0.5, mask_image, 1 - 0.5, gamma=0
        )

        return scene
