import cv2
# import numpy as np
# import logging
# import torch
# from lightglue import SuperPoint
# from lightglue.utils import numpy_image_to_torch, resize_image
# from pipeline.base import PipelineStage, StageContext
# from pipeline.types import Frame, Features, Mask

# logger = logging.getLogger(__name__)

# class FeatureExtractorStage(PipelineStage[Frame, Features]):
#     """Extract SuperPoint features from frames for speed estimation"""
    
#     def __init__(self, max_num_keypoints: int = 2000):
#         super().__init__("FeatureExtractor")
#         self.max_num_keypoints = max_num_keypoints
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
    
#     async def process(self, frame: Frame, ctx: StageContext) -> Features | None:
#         try:
#             frame_tensor = numpy_image_to_torch(frame.data).to(self.device)
            
#             with torch.no_grad():
#                 features = self.extractor.extract(frame_tensor, resize=None)
            
#             keypoints = features["keypoints"][0].cpu().numpy()
#             descriptors = features["descriptors"][0].cpu().numpy()
            
#             if len(keypoints) < 10:
#                 logger.warning(f"Frame {frame.index}: Only {len(keypoints)} keypoints found")
#                 return None
            
#             return Features(
#                 keypoints=keypoints,
#                 descriptors=descriptors,
#                 frame_index=frame.index
#             )
            
#         except Exception as e:
#             logger.error(f"Frame {frame.index}: Error - {e}")
#             return None


# class MaskedFeatureExtractorStage(PipelineStage[tuple[Frame, Mask], Features]):
#     """Extract SuperPoint features only from static regions using mask"""
    
#     def __init__(self, max_num_keypoints: int = 2000):
#         super().__init__("MaskedFeatureExtractor")
#         self.max_num_keypoints = max_num_keypoints
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
    
#     async def process(self, data: tuple[Frame, Mask], ctx: StageContext) -> Features | None:
#         frame, mask = data
        
#         try:
#             frame_tensor = numpy_image_to_torch(frame.data).to(self.device)
            
#             with torch.no_grad():
#                 features = self.extractor.extract(frame_tensor, resize=None)
            
#             keypoints = features["keypoints"][0].cpu().numpy()
#             descriptors = features["descriptors"][0].cpu().numpy()
            
#             # Filter keypoints that fall in dynamic regions
#             valid_indices = []
#             for i, kp in enumerate(keypoints):
#                 x, y = int(kp[0]), int(kp[1])
#                 if 0 <= x < mask.data.shape[1] and 0 <= y < mask.data.shape[0]:
#                     if mask.data[y, x] > 128: 
#                         valid_indices.append(i)
            
#             if not valid_indices:
#                 logger.warning(f"Frame {frame.index}: No valid keypoints after masking")
#                 return None
                
#             keypoints = keypoints[valid_indices]
#             descriptors = descriptors[valid_indices]
            
#             if len(keypoints) < 10:
#                 logger.warning(f"Frame {frame.index}: Only {len(keypoints)} keypoints after masking")
#                 return None
            
#             return Features(
#                 keypoints=keypoints,
#                 descriptors=descriptors,
#                 frame_index=frame.index
#             )
            
#         except Exception as e:
#             logger.error(f"Frame {frame.index}: Error - {e}")
#             return None