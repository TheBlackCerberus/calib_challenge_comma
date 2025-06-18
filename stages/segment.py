from pipeline.base import PipelineStage, StageContext
from pipeline.types import Frame, Mask
import numpy as np
from ultralytics import YOLO
import cv2
import logging

logger = logging.getLogger(__name__)

class DynamicMaskingStage(PipelineStage[Frame, Mask]):
    """Apply YOLO-NAS to mask moving objects before feature extraction"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        super().__init__("DynamicMasking")
        self.model = YOLO("yolo11n-seg.pt")
        self.confidence_threshold = confidence_threshold
        self.dynamic_class_ids = [0, 1, 2, 3, 5, 7]
        
    
    async def process(self, frame: Frame, ctx: StageContext) -> tuple[Frame, Mask]:
        results = self.model.predict(frame.data, conf=self.confidence_threshold, classes=self.dynamic_class_ids)
        
        h, w = frame.data.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        for result in results:
            if result.boxes is not None and result.masks is not None:
                for box, seg_mask in zip(result.boxes.data, result.masks.data):
                    if int(box[5]) in self.dynamic_class_ids:
                        binary_mask = seg_mask.cpu().numpy().astype(np.uint8)
                        if binary_mask.shape != (h, w):
                            binary_mask = cv2.resize(binary_mask, (w, h))
                        mask[binary_mask > 0.5] = 0

        logger.info(f"Dynamic mask: {mask.shape}")
        
        return frame, Mask(data=mask) 