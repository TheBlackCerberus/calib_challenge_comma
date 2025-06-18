from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass
class Frame:
    """Single video frame with metadata"""
    data: NDArray[np.uint8]
    index: int
    timestamp: float
    speed: float | None = None

@dataclass
class Mask:
    """Mask of a frame"""
    data: NDArray[np.uint8]

@dataclass
class Features:
    """Extracted features from a frame"""
    keypoints: NDArray[np.float32]
    descriptors: NDArray[np.uint8]
    frame_index: int

@dataclass
class MotionEstimate:
    """Estimated motion between frames"""
    pitch: float
    yaw: float
    confidence: float
    frame_index: int
    inlier_ratio: float | None = None

@dataclass
class CalibrationResult:
    """Final calibration radians"""
    pitch: float
    yaw: float
    frame_index: int

@dataclass
class SpeedEstimate:
    """Estimated speed in m/s"""
    speed_ms: float
    frame_index: int