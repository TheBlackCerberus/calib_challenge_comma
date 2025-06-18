import cv2
import asyncio
from typing import AsyncIterator
from pipeline.base import SourceStage, StageContext
from pipeline.types import Frame
import logging

logger = logging.getLogger(__name__)

class VideoReaderStage(SourceStage[Frame]):
    """Reads frames from video file"""
    
    def __init__(self, speed_data: list[float] | None = None):
        super().__init__("VideoReader")
        self.speed_data = speed_data
        
    async def generate(self, ctx: StageContext) -> AsyncIterator[Frame]:
        """Generate frames from video"""
        cap = cv2.VideoCapture(ctx.video_path)
        frame_index = 0
        
        try:
            while cap.isOpened():
                ret, frame_data = cap.read()
                if not ret:
                    break
                    
                speed = None
                
                # Create frame object
                frame = Frame(
                    data=frame_data,
                    index=frame_index,
                    timestamp=frame_index / 20.0,  # 20 fps
                    speed=speed
                )
                yield frame

                logger.info(f"Frame {frame_index}: Speed: {speed}")
                frame_index += 1
                
        finally:
            cap.release()