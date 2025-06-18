import cv2
import numpy as np
from pipeline.base import PipelineStage, StageContext
from pipeline.types import Features, MotionEstimate
import logging

logger = logging.getLogger(__name__)

class MotionEstimatorStage(PipelineStage[Features, MotionEstimate]):
    """Estimates camera motion from features - MINIMAL VERSION"""
    
    def __init__(self):
        super().__init__("MotionEstimator")
        self.prev_features: Features | None = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
    async def process(self, features: Features, ctx: StageContext) -> MotionEstimate | None:
        """Estimate motion between CONSECUTIVE frames"""
        
        # First frame - return zero motion
        if self.prev_features is None:
            self.prev_features = features
            logger.info(f"Frame {features.frame_index}: First frame - zero motion")
            
            return MotionEstimate(
                pitch=0.0,
                yaw=0.0,
                confidence=0.0,
                frame_index=features.frame_index,
                inlier_ratio=0.0
            )
        
        matches = self.matcher.knnMatch(
            self.prev_features.descriptors,
            features.descriptors,
            k=2
        )
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.8 * n.distance:
                    good_matches.append(m)
        
        logger.info(f"Frame {features.frame_index}: {len(good_matches)} good matches")
        
        if len(good_matches) < 8:
            logger.warning(f"Frame {features.frame_index}: Too few matches")
            self.prev_features = features
            return MotionEstimate(
                pitch=0.0,
                yaw=0.0,
                confidence=0.0,
                frame_index=features.frame_index,
                inlier_ratio=0.0
            )
        
        pts1 = np.array([self.prev_features.keypoints[m.queryIdx] for m in good_matches])
        pts2 = np.array([features.keypoints[m.trainIdx] for m in good_matches])
        
        try:
            E, mask = cv2.findEssentialMat(
                pts1, pts2,
                ctx.camera_matrix,
                method=cv2.LMEDS,
                prob=0.99
            )
            
            _, R, t, mask_pose = cv2.recoverPose(
                E, pts1, pts2, 
                ctx.camera_matrix,
                mask=mask
            )
            
            t_normalized = t.flatten()
            
            if np.linalg.norm(t_normalized) < 1e-6:
                pitch = 0.0
                yaw = 0.0
            else:
                # Extract angles from epipole direction
                angles = np.arccos(np.clip(t_normalized, -1.0, 1.0))
                yaw = -(angles[2] - np.pi)
                pitch = -(angles[1] - np.pi / 2)
            
            logger.info(f"Frame {features.frame_index}: pitch={pitch:.6f}, yaw={yaw:.6f}")
            
            self.prev_features = features
            
            inlier_ratio = float(np.sum(mask_pose) / len(mask_pose)) if len(mask_pose) > 0 else 0.0
            
            return MotionEstimate(
                pitch=pitch,
                yaw=yaw,
                confidence=inlier_ratio,
                frame_index=features.frame_index,
                inlier_ratio=inlier_ratio
            )
            
        except Exception as e:
            logger.error(f"Frame {features.frame_index}: Error - {str(e)}")
            
            self.prev_features = features
            
            return MotionEstimate(
                pitch=0.0,
                yaw=0.0,
                confidence=0.0,
                frame_index=features.frame_index,
                inlier_ratio=0.0
            )