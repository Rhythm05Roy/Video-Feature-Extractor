"""Video Feature Extractor - Enterprise-grade video analysis tool.

This package provides comprehensive video feature extraction capabilities including:
- Shot cut detection using frame-to-frame analysis
- Motion analysis via optical flow
- Text detection (OCR) using Tesseract
- Object/Person detection using YOLO

Example usage:
    from video_feature_extractor import VideoFeatureExtractor, ExtractorConfig
    
    config = ExtractorConfig()
    extractor = VideoFeatureExtractor(config)
    results = extractor.extract("video.mp4", features=["cuts", "motion"])
"""

__version__ = "2.0.0"
__author__ = "Video Feature Extractor Team"

from video_feature_extractor.core import VideoFeatureExtractor
from video_feature_extractor.config import ExtractorConfig
from video_feature_extractor.exceptions import (
    VideoFeatureExtractorError,
    VideoNotFoundError,
    VideoOpenError,
    ModelNotFoundError,
    OCRError,
    InvalidFeatureError,
)

__all__ = [
    "VideoFeatureExtractor",
    "ExtractorConfig",
    "VideoFeatureExtractorError",
    "VideoNotFoundError",
    "VideoOpenError",
    "ModelNotFoundError",
    "OCRError",
    "InvalidFeatureError",
    "__version__",
]
