"""Feature extractor modules for Video Feature Extractor."""

from video_feature_extractor.extractors.base import BaseExtractor, ProgressCallback
from video_feature_extractor.extractors.shot_cuts import ShotCutExtractor
from video_feature_extractor.extractors.motion import MotionExtractor
from video_feature_extractor.extractors.text_ocr import TextOCRExtractor
from video_feature_extractor.extractors.object_detection import ObjectDetectionExtractor

__all__ = [
    "BaseExtractor",
    "ProgressCallback",
    "ShotCutExtractor",
    "MotionExtractor",
    "TextOCRExtractor",
    "ObjectDetectionExtractor",
]

# Feature name to extractor class mapping
EXTRACTOR_MAP = {
    "cuts": ShotCutExtractor,
    "motion": MotionExtractor,
    "text": TextOCRExtractor,
    "objects": ObjectDetectionExtractor,
}
