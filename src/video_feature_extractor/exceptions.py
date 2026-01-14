"""Custom exception hierarchy for Video Feature Extractor.

This module defines a structured exception hierarchy for better error handling
and debugging in enterprise environments.
"""


class VideoFeatureExtractorError(Exception):
    """Base exception for all Video Feature Extractor errors.
    
    All custom exceptions in this package inherit from this class,
    allowing for easy catching of any extractor-related error.
    
    Attributes:
        message: Human-readable error description.
        details: Optional dictionary with additional error context.
    """
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class VideoNotFoundError(VideoFeatureExtractorError):
    """Raised when the specified video file does not exist.
    
    Example:
        >>> raise VideoNotFoundError("/path/to/missing.mp4")
    """
    
    def __init__(self, video_path: str):
        super().__init__(
            f"Video file not found: {video_path}",
            details={"video_path": video_path}
        )
        self.video_path = video_path


class VideoOpenError(VideoFeatureExtractorError):
    """Raised when a video file exists but cannot be opened.
    
    This typically occurs when:
    - The file is corrupted
    - The codec is not supported
    - The file is not a valid video format
    """
    
    def __init__(self, video_path: str, reason: str = None):
        message = f"Unable to open video: {video_path}"
        if reason:
            message += f" - {reason}"
        super().__init__(message, details={"video_path": video_path, "reason": reason})
        self.video_path = video_path
        self.reason = reason


class ModelNotFoundError(VideoFeatureExtractorError):
    """Raised when a required ML model file is not found.
    
    This applies to YOLO weights, config files, or class name files.
    """
    
    def __init__(self, model_type: str, model_path: str):
        super().__init__(
            f"{model_type} model not found: {model_path}",
            details={"model_type": model_type, "model_path": model_path}
        )
        self.model_type = model_type
        self.model_path = model_path


class OCRError(VideoFeatureExtractorError):
    """Raised when OCR processing fails.
    
    Common causes:
    - Tesseract not installed
    - Tesseract not in PATH
    - Invalid image format
    """
    
    def __init__(self, message: str, tesseract_installed: bool = None):
        super().__init__(
            message,
            details={"tesseract_installed": tesseract_installed}
        )
        self.tesseract_installed = tesseract_installed


class InvalidFeatureError(VideoFeatureExtractorError):
    """Raised when an invalid feature name is requested.
    
    Attributes:
        requested_features: List of features that were requested.
        valid_features: Set of valid feature names.
    """
    
    def __init__(self, requested_features: list, valid_features: set):
        invalid = set(requested_features) - valid_features
        super().__init__(
            f"Invalid feature(s) requested: {invalid}. Valid features: {valid_features}",
            details={"invalid_features": list(invalid), "valid_features": list(valid_features)}
        )
        self.requested_features = requested_features
        self.valid_features = valid_features


class ConfigurationError(VideoFeatureExtractorError):
    """Raised when configuration is invalid or cannot be loaded."""
    
    def __init__(self, message: str, config_path: str = None):
        super().__init__(message, details={"config_path": config_path})
        self.config_path = config_path


class ExtractionError(VideoFeatureExtractorError):
    """Raised when feature extraction fails during processing.
    
    This is a general error for runtime extraction failures.
    """
    
    def __init__(self, feature: str, message: str, frame_number: int = None):
        super().__init__(
            f"Extraction failed for '{feature}': {message}",
            details={"feature": feature, "frame_number": frame_number}
        )
        self.feature = feature
        self.frame_number = frame_number
