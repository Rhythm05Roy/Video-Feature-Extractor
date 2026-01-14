"""Object and person detection extractor using YOLOv8.

Detects objects and people in video frames using the ultralytics YOLOv8 model.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import cv2
import numpy as np

from video_feature_extractor.extractors.base import BaseExtractor, ProgressCallback
from video_feature_extractor.config import ExtractorConfig, ObjectDetectionConfig
from video_feature_extractor.exceptions import VideoOpenError, ModelNotFoundError

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


class ObjectDetectionExtractor(BaseExtractor):
    """Extractor for detecting objects and people using YOLOv8.
    
    Uses the ultralytics YOLOv8 model to detect objects and categorize
    them as either 'person' or 'object', calculating dominance ratios.
    
    Example:
        >>> config = ExtractorConfig()
        >>> extractor = ObjectDetectionExtractor(config)
        >>> results = extractor.extract(Path("video.mp4"))
        >>> print(results["dominant_category"])
        'person'
    """
    
    name = "Object/Person Detection"
    feature_key = "object_person_dominance"
    
    # COCO class ID for person
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        config: ExtractorConfig,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(config, logger, progress_callback)
        self._config = config.object_detection
        self._model: Optional[Any] = None
    
    def get_config_section(self) -> ObjectDetectionConfig:
        return self._config
    
    def is_available(self) -> bool:
        """Check if YOLO is available."""
        return YOLO_AVAILABLE
    
    def _load_model(self) -> Any:
        """Load the YOLO model.
        
        Returns:
            Loaded YOLO model instance.
            
        Raises:
            ModelNotFoundError: If model cannot be loaded.
        """
        if self._model is not None:
            return self._model
        
        if not YOLO_AVAILABLE:
            raise ModelNotFoundError(
                "YOLO",
                "ultralytics package not installed. Install with: pip install ultralytics"
            )
        
        model_name = f"yolov8{self._config.model_size}.pt"
        self.logger.info(f"Loading YOLO model: {model_name}")
        
        try:
            self._model = YOLO(model_name)
            
            # Set device
            if self._config.use_gpu:
                self.logger.debug("Using GPU for inference")
            else:
                self.logger.debug("Using CPU for inference")
            
            return self._model
        except Exception as e:
            raise ModelNotFoundError("YOLO", str(e))
    
    def extract(self, video_path: Path) -> Dict[str, Any]:
        """Detect objects and people in video frames.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary with:
                - persons_detected: Total person detections
                - objects_detected: Total non-person object detections
                - person_ratio: Ratio of person detections
                - object_ratio: Ratio of object detections
                - dominant_category: 'person', 'object', or 'tie'
                - class_distribution: Count of each detected class
                - frames_evaluated: Number of frames processed
                - frame_step_used: Actual frame step used
                - confidence_threshold: Detection confidence used
                - model_used: YOLO model identifier
                
        Raises:
            VideoOpenError: If the video cannot be opened.
            ModelNotFoundError: If YOLO model cannot be loaded.
        """
        if not self.is_available():
            self.logger.warning("YOLO not available, skipping object detection")
            return {
                "persons_detected": 0,
                "objects_detected": 0,
                "person_ratio": 0.0,
                "object_ratio": 0.0,
                "dominant_category": "unknown",
                "frames_evaluated": 0,
                "error": "ultralytics package not installed",
                "available": False,
            }
        
        self.logger.info(f"Starting object detection on: {video_path}")
        
        # Load model
        model = self._load_model()
        
        cfg = self._config
        capture = cv2.VideoCapture(str(video_path))
        
        if not capture.isOpened():
            raise VideoOpenError(str(video_path))
        
        try:
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            estimated_samples = total_frames // cfg.frame_step
            self.logger.debug(
                f"Video has {total_frames} frames, "
                f"sampling ~{estimated_samples} frames with step={cfg.frame_step}"
            )
            
            persons = 0
            objects = 0
            class_counts: Dict[str, int] = {}
            frames_evaluated = 0
            
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                
                frames_evaluated += 1
                
                # Run YOLO inference
                results = model(
                    frame, 
                    conf=cfg.confidence_threshold,
                    verbose=False,
                    device="cuda" if cfg.use_gpu else "cpu"
                )
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        
                        # Update class distribution
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Categorize as person or object
                        if class_id == self.PERSON_CLASS_ID:
                            persons += 1
                        else:
                            objects += 1
                
                # Report progress
                if frames_evaluated % 5 == 0:
                    self.on_progress(
                        frames_evaluated, 
                        estimated_samples, 
                        f"p={persons}, o={objects}"
                    )
                
                # Frame stepping
                if cfg.frame_step > 1:
                    for _ in range(cfg.frame_step - 1):
                        capture.grab()
            
            self.on_progress(estimated_samples, estimated_samples, "Complete")
            
        finally:
            capture.release()
        
        # Calculate ratios
        total = persons + objects
        person_ratio = persons / total if total > 0 else 0.0
        object_ratio = objects / total if total > 0 else 0.0
        
        # Determine dominance
        if persons > objects:
            dominant = "person"
        elif objects > persons:
            dominant = "object"
        else:
            dominant = "tie"
        
        # Sort class distribution by count
        sorted_classes = dict(
            sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        )
        
        self.logger.info(
            f"Object detection complete: {persons} persons, {objects} objects, "
            f"dominant={dominant}"
        )
        
        return {
            "persons_detected": persons,
            "objects_detected": objects,
            "person_ratio": round(person_ratio, 4),
            "object_ratio": round(object_ratio, 4),
            "dominant_category": dominant,
            "class_distribution": sorted_classes,
            "top_classes": list(sorted_classes.keys())[:10],
            "frames_evaluated": frames_evaluated,
            "frame_step_used": cfg.frame_step,
            "confidence_threshold": cfg.confidence_threshold,
            "nms_threshold": cfg.nms_threshold,
            "model_used": f"yolov8{cfg.model_size}",
            "available": True,
        }
