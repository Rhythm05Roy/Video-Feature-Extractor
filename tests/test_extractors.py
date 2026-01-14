"""Unit tests for feature extractors."""

from pathlib import Path

import pytest

from video_feature_extractor.config import ExtractorConfig
from video_feature_extractor.extractors import (
    ShotCutExtractor,
    MotionExtractor,
    TextOCRExtractor,
    ObjectDetectionExtractor,
)
from video_feature_extractor.exceptions import VideoOpenError


class TestShotCutExtractor:
    """Tests for shot cut detection."""
    
    def test_is_available(self, config: ExtractorConfig):
        """Shot cut extractor should always be available."""
        extractor = ShotCutExtractor(config)
        assert extractor.is_available() is True
    
    def test_extract_generated_video(self, config: ExtractorConfig, generated_video: Path):
        """Should detect scene cuts in generated video."""
        extractor = ShotCutExtractor(config)
        result = extractor.extract(generated_video)
        
        assert "shot_cut_count" in result
        assert isinstance(result["shot_cut_count"], int)
        assert result["shot_cut_count"] >= 0
        assert "frame_step_used" in result
        assert "mean_diff_threshold" in result
    
    def test_extract_sample_video(self, config: ExtractorConfig, sample_video_path: Path):
        """Should extract shot cuts from sample video if it exists."""
        if not sample_video_path.is_file():
            pytest.skip("Sample video not found")
        
        extractor = ShotCutExtractor(config)
        result = extractor.extract(sample_video_path)
        
        assert "shot_cut_count" in result
        assert result["shot_cut_count"] >= 0
    
    def test_extract_invalid_video(self, config: ExtractorConfig, invalid_video: Path):
        """Should raise VideoOpenError for invalid video."""
        extractor = ShotCutExtractor(config)
        
        with pytest.raises(VideoOpenError):
            extractor.extract(invalid_video)


class TestMotionExtractor:
    """Tests for motion analysis."""
    
    def test_is_available(self, config: ExtractorConfig):
        """Motion extractor should always be available."""
        extractor = MotionExtractor(config)
        assert extractor.is_available() is True
    
    def test_extract_generated_video(self, config: ExtractorConfig, generated_video: Path):
        """Should analyze motion in generated video."""
        extractor = MotionExtractor(config)
        result = extractor.extract(generated_video)
        
        assert "average_motion_magnitude" in result
        assert isinstance(result["average_motion_magnitude"], float)
        assert result["average_motion_magnitude"] >= 0
        assert "motion_samples" in result
        assert result["motion_samples"] > 0
    
    def test_extract_sample_video(self, config: ExtractorConfig, sample_video_path: Path):
        """Should analyze motion in sample video if it exists."""
        if not sample_video_path.is_file():
            pytest.skip("Sample video not found")
        
        extractor = MotionExtractor(config)
        result = extractor.extract(sample_video_path)
        
        assert "average_motion_magnitude" in result
        assert "max_motion_magnitude" in result
        assert "min_motion_magnitude" in result


class TestTextOCRExtractor:
    """Tests for text detection."""
    
    def test_availability(self, config: ExtractorConfig):
        """Should report availability based on Tesseract installation."""
        extractor = TextOCRExtractor(config)
        # Just check that it returns a boolean
        assert isinstance(extractor.is_available(), bool)
    
    def test_extract_returns_valid_structure(self, config: ExtractorConfig, generated_video: Path):
        """Should return valid output structure even if OCR unavailable."""
        extractor = TextOCRExtractor(config)
        result = extractor.extract(generated_video)
        
        assert "text_present_ratio" in result
        assert isinstance(result["text_present_ratio"], float)
        assert 0 <= result["text_present_ratio"] <= 1
    
    def test_extract_video_with_text(self, config: ExtractorConfig, video_with_text: Path):
        """Should detect text if Tesseract is available."""
        extractor = TextOCRExtractor(config)
        
        if not extractor.is_available():
            pytest.skip("Tesseract not available")
        
        result = extractor.extract(video_with_text)
        
        assert result["text_present_ratio"] > 0
        assert result["frames_with_text"] > 0


class TestObjectDetectionExtractor:
    """Tests for object/person detection."""
    
    def test_availability(self, config: ExtractorConfig):
        """Should report availability based on ultralytics installation."""
        extractor = ObjectDetectionExtractor(config)
        # ultralytics should be available
        assert extractor.is_available() is True
    
    def test_extract_generated_video(self, config: ExtractorConfig, generated_video: Path):
        """Should process generated video."""
        extractor = ObjectDetectionExtractor(config)
        
        if not extractor.is_available():
            pytest.skip("YOLO not available")
        
        result = extractor.extract(generated_video)
        
        assert "persons_detected" in result
        assert "objects_detected" in result
        assert "dominant_category" in result
        assert result["dominant_category"] in ["person", "object", "tie", "unknown"]
    
    def test_extract_sample_video(self, config: ExtractorConfig, sample_video_path: Path):
        """Should detect objects in sample video if both exist."""
        if not sample_video_path.is_file():
            pytest.skip("Sample video not found")
        
        extractor = ObjectDetectionExtractor(config)
        
        if not extractor.is_available():
            pytest.skip("YOLO not available")
        
        result = extractor.extract(sample_video_path)
        
        assert "frames_evaluated" in result
        assert result["frames_evaluated"] > 0
