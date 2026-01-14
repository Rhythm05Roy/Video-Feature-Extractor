"""Base extractor class for Video Feature Extractor.

Provides an abstract base class that all feature extractors must implement,
ensuring consistent interface and behavior across the codebase.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import logging

from video_feature_extractor.config import ExtractorConfig


# Type alias for progress callback
ProgressCallback = Callable[[int, int, Optional[str]], None]


class BaseExtractor(ABC):
    """Abstract base class for all feature extractors.
    
    All feature extraction implementations must inherit from this class
    and implement the `extract` method.
    
    Attributes:
        name: Human-readable name of the extractor.
        feature_key: Key used in the output dictionary for this feature.
        logger: Logger instance for this extractor.
        progress_callback: Optional callback for progress updates.
    """
    
    name: str = "Base Extractor"
    feature_key: str = "base"
    
    def __init__(
        self,
        config: ExtractorConfig,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        """Initialize the extractor.
        
        Args:
            config: Extractor configuration.
            logger: Optional logger instance. Creates default if not provided.
            progress_callback: Optional callback for progress updates.
        """
        self.config = config
        self.logger = logger or logging.getLogger(f"video_feature_extractor.{self.feature_key}")
        self.progress_callback = progress_callback
    
    @abstractmethod
    def extract(self, video_path: Path) -> Dict[str, Any]:
        """Extract features from the video.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary containing extracted feature data.
            
        Raises:
            VideoOpenError: If the video cannot be opened.
            ExtractionError: If feature extraction fails.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this extractor is available (dependencies met).
        
        Returns:
            True if the extractor can be used, False otherwise.
        """
        pass
    
    def on_progress(self, current: int, total: int, message: Optional[str] = None) -> None:
        """Report progress to callback if registered.
        
        Args:
            current: Current progress count.
            total: Total items to process.
            message: Optional status message.
        """
        if self.progress_callback:
            self.progress_callback(current, total, message)
        
        # Also log at debug level
        if total > 0:
            pct = (current / total) * 100
            log_msg = f"{self.name}: {current}/{total} ({pct:.1f}%)"
            if message:
                log_msg += f" - {message}"
            self.logger.debug(log_msg)
    
    def get_config_section(self) -> Any:
        """Get the configuration section for this extractor.
        
        Override in subclasses to return the appropriate config section.
        
        Returns:
            Configuration dataclass for this extractor.
        """
        return None
    
    def validate_config(self) -> None:
        """Validate configuration for this extractor.
        
        Override in subclasses to add custom validation.
        
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', feature_key='{self.feature_key}')"
