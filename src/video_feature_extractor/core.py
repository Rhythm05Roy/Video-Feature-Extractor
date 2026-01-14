"""Core orchestration module for Video Feature Extractor.

Provides the main VideoFeatureExtractor class that coordinates
feature extraction across multiple extractors.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import json
import logging

from video_feature_extractor import __version__
from video_feature_extractor.config import ExtractorConfig
from video_feature_extractor.exceptions import (
    InvalidFeatureError,
    VideoNotFoundError,
    VideoFeatureExtractorError,
)
from video_feature_extractor.logging_config import setup_logging, ProgressLogger
from video_feature_extractor.utils.video import get_video_metadata, validate_video_file
from video_feature_extractor.extractors import (
    EXTRACTOR_MAP,
    ShotCutExtractor,
    MotionExtractor,
    TextOCRExtractor,
    ObjectDetectionExtractor,
)


# Type alias for progress callback
ProgressCallback = Callable[[str, int, int, Optional[str]], None]


class VideoFeatureExtractor:
    """Main facade for video feature extraction.
    
    This class orchestrates the extraction of multiple features from
    a video file, coordinating between individual extractors and
    providing a unified interface.
    
    Example:
        >>> from video_feature_extractor import VideoFeatureExtractor, ExtractorConfig
        >>> 
        >>> # Default configuration
        >>> extractor = VideoFeatureExtractor()
        >>> results = extractor.extract("video.mp4", features=["cuts", "motion"])
        >>> 
        >>> # Custom configuration
        >>> config = ExtractorConfig.from_yaml("config.yaml")
        >>> extractor = VideoFeatureExtractor(config)
        >>> results = extractor.extract("video.mp4")
    
    Attributes:
        config: Extractor configuration.
        logger: Logger instance.
        available_features: Set of feature names that can be extracted.
    """
    
    available_features = {"cuts", "motion", "text", "objects"}
    
    def __init__(
        self,
        config: Optional[ExtractorConfig] = None,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """Initialize the video feature extractor.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
            logger: Optional logger. Creates one based on config if not provided.
            progress_callback: Optional callback for progress updates.
                Signature: (feature_name, current, total, message) -> None
        """
        self.config = config or ExtractorConfig()
        self.config.validate()
        
        # Setup logging
        if logger:
            self.logger = logger
        else:
            self.logger = setup_logging(
                level=self.config.logging.level,
                log_file=self.config.logging.log_file,
                json_format=self.config.logging.json_format
            )
        
        self.progress_callback = progress_callback
        self._extractors: Dict[str, Any] = {}
    
    def _get_extractor(self, feature: str):
        """Get or create an extractor instance.
        
        Args:
            feature: Feature name (cuts, motion, text, objects).
            
        Returns:
            Extractor instance.
        """
        if feature not in self._extractors:
            extractor_class = EXTRACTOR_MAP.get(feature)
            if extractor_class:
                # Create progress callback wrapper for this feature
                def feature_progress(current: int, total: int, message: Optional[str] = None):
                    if self.progress_callback:
                        self.progress_callback(feature, current, total, message)
                
                self._extractors[feature] = extractor_class(
                    self.config,
                    logger=self.logger,
                    progress_callback=feature_progress
                )
        
        return self._extractors.get(feature)
    
    def extract(
        self,
        video_path: str | Path,
        features: Optional[List[str]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Extract features from a video file.
        
        Args:
            video_path: Path to the video file.
            features: List of features to extract. If None, uses config defaults.
                Valid features: 'cuts', 'motion', 'text', 'objects'
            include_metadata: Whether to include video metadata in results.
            
        Returns:
            Dictionary containing:
                - video_path: Path to the analyzed video
                - video_metadata: Resolution, FPS, duration, etc. (if include_metadata)
                - extraction_timestamp: ISO format timestamp
                - features_requested: List of requested features
                - results: Dictionary of feature extraction results
                - processing_time_seconds: Total processing time
                - extractor_version: Version of the extractor
                
        Raises:
            VideoNotFoundError: If the video file doesn't exist.
            InvalidFeatureError: If an invalid feature name is requested.
            VideoFeatureExtractorError: For other extraction errors.
        """
        start_time = datetime.now()
        video_path = Path(video_path)
        
        # Validate video exists
        video_path = validate_video_file(video_path)
        
        # Determine features to extract
        if features is None:
            features = self.config.default_features
        
        # Validate feature names
        invalid = set(features) - self.available_features
        if invalid:
            raise InvalidFeatureError(features, self.available_features)
        
        self.logger.info(
            f"Starting extraction of {len(features)} features from: {video_path}"
        )
        
        # Initialize output
        output: Dict[str, Any] = {
            "video_path": str(video_path),
            "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
            "features_requested": features,
            "extractor_version": __version__,
        }
        
        # Add video metadata if requested
        if include_metadata:
            try:
                metadata = get_video_metadata(video_path)
                output["video_metadata"] = metadata.to_dict()
                self.logger.debug(f"Video metadata: {metadata}")
            except Exception as e:
                self.logger.warning(f"Failed to extract metadata: {e}")
                output["video_metadata"] = {"error": str(e)}
        
        # Extract each feature
        results: Dict[str, Any] = {}
        
        for feature in features:
            self.logger.info(f"Extracting feature: {feature}")
            
            extractor = self._get_extractor(feature)
            if extractor is None:
                self.logger.warning(f"No extractor available for: {feature}")
                results[extractor.feature_key if extractor else feature] = {
                    "error": "Extractor not available"
                }
                continue
            
            try:
                feature_result = extractor.extract(video_path)
                results[extractor.feature_key] = feature_result
            except VideoFeatureExtractorError as e:
                self.logger.error(f"Extraction failed for {feature}: {e}")
                results[extractor.feature_key] = {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            except Exception as e:
                self.logger.exception(f"Unexpected error in {feature}")
                results[extractor.feature_key] = {
                    "error": str(e),
                    "error_type": "UnexpectedError"
                }
        
        output["results"] = results
        
        # Calculate processing time
        elapsed = (datetime.now() - start_time).total_seconds()
        output["processing_time_seconds"] = round(elapsed, 2)
        
        self.logger.info(
            f"Extraction complete in {elapsed:.2f}s - "
            f"{len([r for r in results.values() if 'error' not in r])}/{len(features)} features succeeded"
        )
        
        return output
    
    def extract_to_json(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        features: Optional[List[str]] = None,
        pretty: bool = True
    ) -> str:
        """Extract features and return/save as JSON.
        
        Args:
            video_path: Path to the video file.
            output_path: Optional path to save JSON output.
            features: Features to extract (uses config defaults if None).
            pretty: Whether to pretty-print the JSON.
            
        Returns:
            JSON string of extraction results.
        """
        results = self.extract(video_path, features)
        
        indent = 2 if pretty else None
        json_output = json.dumps(results, indent=indent, default=str)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_output)
            self.logger.info(f"Results saved to: {output_path}")
        
        return json_output
    
    def check_availability(self) -> Dict[str, bool]:
        """Check which features are available.
        
        Returns:
            Dictionary mapping feature names to availability status.
        """
        availability = {}
        
        for feature in self.available_features:
            extractor = self._get_extractor(feature)
            if extractor:
                availability[feature] = extractor.is_available()
            else:
                availability[feature] = False
        
        return availability
