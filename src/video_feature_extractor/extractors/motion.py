"""Motion analysis extractor using optical flow.

Quantifies average motion in video using Farneback optical flow algorithm.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import cv2
import numpy as np

from video_feature_extractor.extractors.base import BaseExtractor, ProgressCallback
from video_feature_extractor.config import ExtractorConfig, MotionConfig
from video_feature_extractor.exceptions import VideoOpenError


class MotionExtractor(BaseExtractor):
    """Extractor for analyzing motion in video using optical flow.
    
    Computes dense optical flow between consecutive frames using the
    Farneback algorithm and calculates motion magnitude statistics.
    
    Example:
        >>> config = ExtractorConfig()
        >>> extractor = MotionExtractor(config)
        >>> results = extractor.extract(Path("video.mp4"))
        >>> print(results["average_motion_magnitude"])
        0.842
    """
    
    name = "Motion Analysis"
    feature_key = "motion_analysis"
    
    def __init__(
        self,
        config: ExtractorConfig,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(config, logger, progress_callback)
        self._config = config.motion
    
    def get_config_section(self) -> MotionConfig:
        return self._config
    
    def is_available(self) -> bool:
        """Motion analysis is always available (only requires OpenCV)."""
        return True
    
    def extract(self, video_path: Path) -> Dict[str, Any]:
        """Analyze motion in the video using optical flow.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary with:
                - average_motion_magnitude: Mean motion across all samples
                - max_motion_magnitude: Maximum motion detected
                - min_motion_magnitude: Minimum motion detected
                - motion_std: Standard deviation of motion
                - motion_samples: Number of frame pairs analyzed
                - frame_step_used: Actual frame step used
                
        Raises:
            VideoOpenError: If the video cannot be opened.
        """
        self.logger.info(f"Starting motion analysis on: {video_path}")
        
        cfg = self._config
        capture = cv2.VideoCapture(str(video_path))
        
        if not capture.isOpened():
            raise VideoOpenError(str(video_path))
        
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.debug(
                f"Video has {total_frames} frames, using step={cfg.frame_step}, "
                f"winsize={cfg.winsize}, levels={cfg.levels}"
            )
            
            magnitudes: List[float] = []
            prev_gray: Optional[np.ndarray] = None
            frame_count = 0
            
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        gray,
                        None,
                        cfg.pyr_scale,
                        cfg.levels,
                        cfg.winsize,
                        cfg.iterations,
                        cfg.poly_n,
                        cfg.poly_sigma,
                        0,
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    magnitudes.append(float(np.mean(mag)))
                
                prev_gray = gray
                frame_count += 1
                
                # Report progress every 100 frames
                if frame_count % 100 == 0:
                    avg_so_far = np.mean(magnitudes) if magnitudes else 0.0
                    self.on_progress(frame_count, total_frames, f"avg={avg_so_far:.3f}")
                
                # Frame stepping
                if cfg.frame_step > 1:
                    for _ in range(cfg.frame_step - 1):
                        capture.grab()
            
            self.on_progress(total_frames, total_frames, "Complete")
            
        finally:
            capture.release()
        
        # Calculate statistics
        if magnitudes:
            avg_motion = float(np.mean(magnitudes))
            max_motion = float(np.max(magnitudes))
            min_motion = float(np.min(magnitudes))
            std_motion = float(np.std(magnitudes))
        else:
            avg_motion = max_motion = min_motion = std_motion = 0.0
        
        self.logger.info(
            f"Motion analysis complete: avg={avg_motion:.3f}, "
            f"samples={len(magnitudes)}"
        )
        
        return {
            "average_motion_magnitude": round(avg_motion, 4),
            "max_motion_magnitude": round(max_motion, 4),
            "min_motion_magnitude": round(min_motion, 4),
            "motion_std": round(std_motion, 4),
            "motion_samples": len(magnitudes),
            "frame_step_used": cfg.frame_step,
        }
