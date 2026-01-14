"""Configuration management for Video Feature Extractor.

This module provides dataclass-based configuration with support for:
- YAML/JSON configuration files
- Environment variable overrides
- Sensible defaults with validation
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any, Dict
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from video_feature_extractor.exceptions import ConfigurationError


@dataclass
class ShotCutConfig:
    """Configuration for shot cut detection."""
    enabled: bool = True
    frame_step: int = 1
    diff_threshold: float = 30.0
    min_gap_frames: int = 5


@dataclass
class MotionConfig:
    """Configuration for motion analysis."""
    enabled: bool = True
    frame_step: int = 2
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2


@dataclass
class TextDetectionConfig:
    """Configuration for OCR text detection."""
    enabled: bool = True
    frame_step: int = 15
    min_confidence: float = 70.0
    language: str = "eng"


@dataclass
class ObjectDetectionConfig:
    """Configuration for YOLO object/person detection."""
    enabled: bool = True
    frame_step: int = 15
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    model_size: str = "n"  # YOLOv8 model size: n, s, m, l, x
    use_gpu: bool = False


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    log_file: Optional[str] = None
    json_format: bool = False
    show_progress: bool = True


@dataclass
class OutputConfig:
    """Configuration for output formatting."""
    format: str = "json"  # json, csv, yaml
    pretty_print: bool = True
    include_metadata: bool = True
    include_timestamps: bool = True


@dataclass
class ExtractorConfig:
    """Main configuration container for Video Feature Extractor.
    
    This dataclass holds all configuration options and provides methods
    for loading from files and environment variables.
    
    Example:
        >>> config = ExtractorConfig()
        >>> config = ExtractorConfig.from_yaml("config.yaml")
        >>> config = ExtractorConfig.from_dict({"logging": {"level": "DEBUG"}})
    """
    
    shot_cut: ShotCutConfig = field(default_factory=ShotCutConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    text_detection: TextDetectionConfig = field(default_factory=TextDetectionConfig)
    object_detection: ObjectDetectionConfig = field(default_factory=ObjectDetectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # General settings
    default_features: List[str] = field(default_factory=lambda: ["cuts", "motion", "text", "objects"])
    cache_results: bool = False
    cache_dir: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractorConfig":
        """Create configuration from a dictionary.
        
        Args:
            data: Dictionary with configuration values.
            
        Returns:
            ExtractorConfig instance with values from the dictionary.
        """
        config = cls()
        
        if "shot_cut" in data:
            config.shot_cut = ShotCutConfig(**data["shot_cut"])
        if "motion" in data:
            config.motion = MotionConfig(**data["motion"])
        if "text_detection" in data:
            config.text_detection = TextDetectionConfig(**data["text_detection"])
        if "object_detection" in data:
            config.object_detection = ObjectDetectionConfig(**data["object_detection"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "output" in data:
            config.output = OutputConfig(**data["output"])
        if "default_features" in data:
            config.default_features = data["default_features"]
        if "cache_results" in data:
            config.cache_results = data["cache_results"]
        if "cache_dir" in data:
            config.cache_dir = data["cache_dir"]
            
        return config
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExtractorConfig":
        """Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            ExtractorConfig instance.
            
        Raises:
            ConfigurationError: If the file cannot be read or parsed.
        """
        if not YAML_AVAILABLE:
            raise ConfigurationError(
                "YAML support requires 'pyyaml' package. Install with: pip install pyyaml",
                config_path=str(path)
            )
        
        path = Path(path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}", config_path=str(path))
        
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data or {})
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax: {e}", config_path=str(path))
    
    @classmethod
    def from_json(cls, path: str | Path) -> "ExtractorConfig":
        """Load configuration from a JSON file.
        
        Args:
            path: Path to the JSON configuration file.
            
        Returns:
            ExtractorConfig instance.
            
        Raises:
            ConfigurationError: If the file cannot be read or parsed.
        """
        path = Path(path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}", config_path=str(path))
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON syntax: {e}", config_path=str(path))
    
    @classmethod
    def from_file(cls, path: str | Path) -> "ExtractorConfig":
        """Load configuration from a file (auto-detects format).
        
        Args:
            path: Path to the configuration file (.yaml, .yml, or .json).
            
        Returns:
            ExtractorConfig instance.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif suffix == ".json":
            return cls.from_json(path)
        else:
            raise ConfigurationError(
                f"Unsupported configuration format: {suffix}. Use .yaml, .yml, or .json",
                config_path=str(path)
            )
    
    def apply_env_overrides(self) -> "ExtractorConfig":
        """Apply environment variable overrides.
        
        Environment variables follow the pattern:
        VFE_<SECTION>_<OPTION>=value
        
        Examples:
            VFE_LOGGING_LEVEL=DEBUG
            VFE_SHOT_CUT_DIFF_THRESHOLD=25.0
            VFE_OBJECT_DETECTION_USE_GPU=true
        
        Returns:
            Self with applied overrides.
        """
        # Logging overrides
        if env_val := os.environ.get("VFE_LOGGING_LEVEL"):
            self.logging.level = env_val
        if env_val := os.environ.get("VFE_LOGGING_JSON_FORMAT"):
            self.logging.json_format = env_val.lower() in ("true", "1", "yes")
        
        # Shot cut overrides
        if env_val := os.environ.get("VFE_SHOT_CUT_DIFF_THRESHOLD"):
            self.shot_cut.diff_threshold = float(env_val)
        if env_val := os.environ.get("VFE_SHOT_CUT_FRAME_STEP"):
            self.shot_cut.frame_step = int(env_val)
        
        # Object detection overrides
        if env_val := os.environ.get("VFE_OBJECT_DETECTION_USE_GPU"):
            self.object_detection.use_gpu = env_val.lower() in ("true", "1", "yes")
        if env_val := os.environ.get("VFE_OBJECT_DETECTION_MODEL_SIZE"):
            self.object_detection.model_size = env_val
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        from dataclasses import asdict
        return asdict(self)
    
    def validate(self) -> None:
        """Validate configuration values.
        
        Raises:
            ConfigurationError: If any configuration value is invalid.
        """
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.logging.level.upper() not in valid_log_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.logging.level}. Must be one of {valid_log_levels}"
            )
        
        if self.shot_cut.frame_step < 1:
            raise ConfigurationError("shot_cut.frame_step must be >= 1")
        if self.motion.frame_step < 1:
            raise ConfigurationError("motion.frame_step must be >= 1")
        if self.text_detection.frame_step < 1:
            raise ConfigurationError("text_detection.frame_step must be >= 1")
        if self.object_detection.frame_step < 1:
            raise ConfigurationError("object_detection.frame_step must be >= 1")
        
        valid_model_sizes = {"n", "s", "m", "l", "x"}
        if self.object_detection.model_size not in valid_model_sizes:
            raise ConfigurationError(
                f"Invalid model_size: {self.object_detection.model_size}. Must be one of {valid_model_sizes}"
            )
        
        valid_features = {"cuts", "motion", "text", "objects"}
        for feature in self.default_features:
            if feature not in valid_features:
                raise ConfigurationError(
                    f"Invalid default feature: {feature}. Must be one of {valid_features}"
                )
