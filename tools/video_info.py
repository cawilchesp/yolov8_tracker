from __future__ import annotations

import cv2
from pathlib import Path

class VideoInfo:
    def __init__(
        self,
        source: str,
        width: int = 0,
        height: int = 0,
        fps: float = 0,
        total_frames: int = None,
        source_type: str = None,
        source_name: str = None
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.total_frames = total_frames
        self.source_type = source_type
        self.source_name = source_name

        self.get_source_info(source)

    @property
    def resolution_wh(self) -> tuple[int, int]:
        return self.width, self.height


    def get_source_info(self, source: str) -> VideoInfo:
        if source.isnumeric():
            self.source_name = "Webcam"
            self.source_type = 'stream'
            video_source = int(source)
        elif source.lower().startswith('rtsp://'):
            self.source_name = "RSTP Stream"
            self.source_type = 'stream'
            video_source = source
        else:
            self.source_name = Path(source).name
            self.source_type = 'file'
            video_source = source

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened(): raise Exception('Source video not available ❌')

        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source_type == 'file' else None
        
        cap.release()
