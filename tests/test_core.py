"""Integration tests for the core VideoFeatureExtractor."""

from pathlib import Path

import pytest

from video_feature_extractor import VideoFeatureExtractor, ExtractorConfig
from video_feature_extractor.exceptions import (
    VideoNotFoundError,
    InvalidFeatureError,
)


class TestVideoFeatureExtractor:
    """Integration tests for VideoFeatureExtractor."""
    
    def test_init_default_config(self):
        """Should initialize with default configuration."""
        extractor = VideoFeatureExtractor()
        assert extractor.config is not None
        assert extractor.logger is not None
    
    def test_init_custom_config(self, config: ExtractorConfig):
        """Should initialize with custom configuration."""
        extractor = VideoFeatureExtractor(config)
        assert extractor.config is config
    
    def test_available_features(self):
        """Should have expected available features."""
        extractor = VideoFeatureExtractor()
        expected = {"cuts", "motion", "text", "objects"}
        assert extractor.available_features == expected
    
    def test_check_availability(self):
        """Should check feature availability."""
        extractor = VideoFeatureExtractor()
        availability = extractor.check_availability()
        
        assert isinstance(availability, dict)
        assert "cuts" in availability
        assert "motion" in availability
        assert "text" in availability
        assert "objects" in availability
        
        # These should always be available (OpenCV-based)
        assert availability["cuts"] is True
        assert availability["motion"] is True
    
    def test_extract_nonexistent_video(self, config: ExtractorConfig, nonexistent_video: Path):
        """Should raise VideoNotFoundError for missing video."""
        extractor = VideoFeatureExtractor(config)
        
        with pytest.raises(VideoNotFoundError):
            extractor.extract(nonexistent_video)
    
    def test_extract_invalid_features(self, config: ExtractorConfig, generated_video: Path):
        """Should raise InvalidFeatureError for unknown features."""
        extractor = VideoFeatureExtractor(config)
        
        with pytest.raises(InvalidFeatureError):
            extractor.extract(generated_video, features=["invalid_feature"])
    
    def test_extract_single_feature(self, config: ExtractorConfig, generated_video: Path):
        """Should extract a single feature."""
        extractor = VideoFeatureExtractor(config)
        result = extractor.extract(generated_video, features=["cuts"])
        
        assert "video_path" in result
        assert "features_requested" in result
        assert result["features_requested"] == ["cuts"]
        assert "results" in result
        assert "shot_cut_detection" in result["results"]
    
    def test_extract_multiple_features(self, config: ExtractorConfig, generated_video: Path):
        """Should extract multiple features."""
        extractor = VideoFeatureExtractor(config)
        result = extractor.extract(generated_video, features=["cuts", "motion"])
        
        assert "cuts" in result["features_requested"]
        assert "motion" in result["features_requested"]
        assert "shot_cut_detection" in result["results"]
        assert "motion_analysis" in result["results"]
    
    def test_extract_includes_metadata(self, config: ExtractorConfig, generated_video: Path):
        """Should include video metadata by default."""
        extractor = VideoFeatureExtractor(config)
        result = extractor.extract(generated_video, features=["cuts"])
        
        assert "video_metadata" in result
        metadata = result["video_metadata"]
        assert "resolution" in metadata
        assert "fps" in metadata
        assert "duration_seconds" in metadata
    
    def test_extract_excludes_metadata(self, config: ExtractorConfig, generated_video: Path):
        """Should exclude metadata when requested."""
        extractor = VideoFeatureExtractor(config)
        result = extractor.extract(
            generated_video, 
            features=["cuts"], 
            include_metadata=False
        )
        
        assert "video_metadata" not in result
    
    def test_extract_includes_timing(self, config: ExtractorConfig, generated_video: Path):
        """Should include processing time."""
        extractor = VideoFeatureExtractor(config)
        result = extractor.extract(generated_video, features=["cuts"])
        
        assert "processing_time_seconds" in result
        assert result["processing_time_seconds"] >= 0  # Can be 0 for fast operations
    
    def test_extract_includes_version(self, config: ExtractorConfig, generated_video: Path):
        """Should include extractor version."""
        extractor = VideoFeatureExtractor(config)
        result = extractor.extract(generated_video, features=["cuts"])
        
        assert "extractor_version" in result
    
    def test_extract_to_json(self, config: ExtractorConfig, generated_video: Path, temp_dir: Path):
        """Should export results to JSON file."""
        extractor = VideoFeatureExtractor(config)
        output_path = temp_dir / "results.json"
        
        json_str = extractor.extract_to_json(
            generated_video,
            output_path=output_path,
            features=["cuts"]
        )
        
        assert output_path.is_file()
        assert len(json_str) > 0
        
        # Verify JSON content
        import json
        data = json.loads(output_path.read_text())
        assert "results" in data
    
    def test_extract_sample_video(self, config: ExtractorConfig, sample_video_path: Path):
        """Should extract all features from sample video."""
        if not sample_video_path.is_file():
            pytest.skip("Sample video not found")
        
        extractor = VideoFeatureExtractor(config)
        result = extractor.extract(
            sample_video_path, 
            features=["cuts", "motion"]  # Skip text/objects for speed
        )
        
        assert "results" in result
        assert len(result["results"]) == 2


class TestConfiguration:
    """Tests for configuration management."""
    
    def test_config_from_dict(self):
        """Should create config from dictionary."""
        data = {
            "shot_cut": {"diff_threshold": 25.0},
            "logging": {"level": "DEBUG"}
        }
        config = ExtractorConfig.from_dict(data)
        
        assert config.shot_cut.diff_threshold == 25.0
        assert config.logging.level == "DEBUG"
    
    def test_config_validation(self):
        """Should validate configuration."""
        config = ExtractorConfig()
        config.validate()  # Should not raise
        
        config.shot_cut.frame_step = 0
        with pytest.raises(Exception):  # ConfigurationError
            config.validate()
    
    def test_config_to_dict(self):
        """Should convert config to dictionary."""
        config = ExtractorConfig()
        data = config.to_dict()
        
        assert isinstance(data, dict)
        assert "shot_cut" in data
        assert "motion" in data
