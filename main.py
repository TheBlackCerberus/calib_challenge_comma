import asyncio
import numpy as np
import logging
import os
from pipeline.executor import Pipeline
from pipeline.base import StageContext
from stages.video_reader import VideoReaderStage
# from stages.feature_extractor import FeatureExtractorStage
# from stages.motion_estimator import MotionEstimatorStage

logging.basicConfig(level=logging.INFO)

async def process_video(video_path: str, output_path: str):
    """Process a single video for calibration"""
    
    # Camera matrix (from comma.ai)
    K = np.array([[910, 0, 582],
                  [0, 910, 437],
                  [0, 0, 1]], dtype=np.float64)
    
    # Create pipeline
    pipeline = Pipeline("SLAM Calibration")
    
    # Add stages
    pipeline.set_source(VideoReaderStage()) \
            # .add_stage(FeatureExtractorStage()) \
            # .add_stage(MotionEstimatorStage()) \
    
    # Create context
    ctx = StageContext(
        video_path=video_path,
        camera_matrix=K
    )
    
    # Run pipeline
    results = await pipeline.run(ctx)
    
    # Save results
    if results:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to numpy array
        angles = np.array([[r.pitch, r.yaw] for r in results])
        np.savetxt(output_path, angles, fmt='%.18e') 
        print(f"Saved {len(results)} calibration estimates to {output_path}")
    else:
        print("No results generated")

async def main():
    """Process all videos"""
    # Process labeled videos
    for i in range(5):
        await process_video(f"data/labeled/{i}.hevc", f"results/{i}.txt")
    
    # Process unlabeled videos
    # for i in range(5, 10):
    #     await process_video(f"unlabeled/{i}.hevc", f"results/{i}.txt")

if __name__ == "__main__":
    asyncio.run(main())