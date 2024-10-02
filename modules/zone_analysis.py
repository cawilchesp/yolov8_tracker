import supervision as sv

import cv2
import json
import torch
import numpy as np
from typing import List

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
    
    def on_zones(self, scene: np.array) -> np.array:
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