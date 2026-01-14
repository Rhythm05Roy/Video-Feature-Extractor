"""Shot cut detection extractor.

Detects hard cuts in video by measuring mean pixel differences between consecutive frames.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import cv2
import numpy as np

from video_feature_extractor.extractors.base import BaseExtractor, ProgressCallback
from video_feature_extractor.config import ExtractorConfig, ShotCutConfig
from video_feature_extractor.exceptions import VideoOpenError


class ShotCutExtractor(BaseExtractor):
    """Extractor for detecting hard cuts (scene changes) in video.
    
    Uses frame-to-frame mean pixel difference to detect abrupt scene changes.
    A cut is registered when the difference exceeds the threshold and is
    separated from the previous cut by at least min_gap_frames.
    
    Example:
        >>> config = ExtractorConfig()
        >>> extractor = ShotCutExtractor(config)
        >>> results = extractor.extract(Path("video.mp4"))
        >>> print(results["shot_cut_count"])
        12
    """
    
    name = "Shot Cut Detection"
    feature_key = "shot_cut_detection"
    
    def __init__(
        self,
        config: ExtractorConfig,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(config, logger, progress_callback)
        self._config = config.shot_cut
    
    def get_config_section(self) -> ShotCutConfig:
        return self._config
    
    def is_available(self) -> bool:
        """Shot cut detection is always available (only requires OpenCV)."""
        return True
    
    def extract(self, video_path: Path) -> Dict[str, Any]:
        """Detect hard cuts in the video.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary with:
                - shot_cut_count: Number of detected cuts
                - cut_frames: List of frame indices where cuts were detected
                - frame_step_used: Actual frame step used
                - mean_diff_threshold: Threshold used for detection
                - min_gap_frames: Minimum gap between cuts
                
        Raises:
            VideoOpenError: If the video cannot be opened.
        """
        self.logger.info(f"Starting shot cut detection on: {video_path}")
        
        cfg = self._config
        capture = cv2.VideoCapture(str(video_path))
        
        if not capture.isOpened():
            raise VideoOpenError(str(video_path))
        
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.debug(f"Video has {total_frames} frames, using step={cfg.frame_step}")
            
            cuts: List[int] = []
            frame_idx = 0
            last_cut_frame = -cfg.min_gap_frames
            prev_gray: Optional[np.ndarray] = None
            
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    frame_diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = float(np.mean(frame_diff))
                    
                    if mean_diff > cfg.diff_threshold and (frame_idx - last_cut_frame) >= cfg.min_gap_frames:
                        cuts.append(frame_idx)
                        last_cut_frame = frame_idx
                        self.logger.debug(f"Cut detected at frame {frame_idx}, diff={mean_diff:.2f}")
                
                prev_gray = gray
                frame_idx += 1
                
                # Report progress every 500 frames
                if frame_idx % 500 == 0:
                    self.on_progress(frame_idx, total_frames, f"{len(cuts)} cuts found")
                
                # Frame stepping
                if cfg.frame_step > 1:
                    for _ in range(cfg.frame_step - 1):
                        capture.grab()
                        frame_idx += 1
            
            self.on_progress(total_frames, total_frames, "Complete")
            
        finally:
            capture.release()
        
        self.logger.info(f"Shot cut detection complete: {len(cuts)} cuts found")
        
        return {
            "shot_cut_count": len(cuts),
            "cut_frames": cuts[:100],  # Limit to first 100 for output size
            "frame_step_used": cfg.frame_step,
            "mean_diff_threshold": cfg.diff_threshold,
            "min_gap_frames": cfg.min_gap_frames,
        }
