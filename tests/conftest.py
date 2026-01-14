"""Pytest fixtures for Video Feature Extractor tests."""

import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest

from video_feature_extractor.config import ExtractorConfig


@pytest.fixture
def config() -> ExtractorConfig:
    """Default configuration for tests."""
    cfg = ExtractorConfig()
    cfg.logging.level = "WARNING"  # Reduce log noise in tests
    return cfg


@pytest.fixture
def sample_video_path() -> Path:
    """Path to the sample video file."""
    return Path(__file__).parent.parent / "src" / "sample-5s.mp4"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def generated_video(temp_dir: Path) -> Path:
    """Generate a synthetic test video.
    
    Creates a short video with known properties for testing:
    - 30 frames at 30 FPS (1 second)
    - 320x240 resolution
    - Alternating colors to simulate scene changes
    """
    video_path = temp_dir / "test_video.mp4"
    
    # Video properties
    fps = 30
    width, height = 320, 240
    num_frames = 30
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    try:
        for i in range(num_frames):
            # Create frame with color based on frame index
            # Change color every 10 frames to simulate scene cuts
            if i < 10:
                color = (255, 0, 0)  # Blue
            elif i < 20:
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red
            
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            
            # Add some motion - moving rectangle
            x = int((i % 10) * 30)
            cv2.rectangle(frame, (x, 100), (x + 50, 150), (255, 255, 255), -1)
            
            writer.write(frame)
    finally:
        writer.release()
    
    return video_path


@pytest.fixture
def video_with_text(temp_dir: Path) -> Path:
    """Generate a video with text for OCR testing."""
    video_path = temp_dir / "text_video.mp4"
    
    fps = 10
    width, height = 640, 480
    num_frames = 10
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    try:
        for i in range(num_frames):
            # White background
            frame = np.full((height, width, 3), 255, dtype=np.uint8)
            
            # Add text
            cv2.putText(
                frame, 
                f"Frame {i}", 
                (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                (0, 0, 0), 
                3
            )
            cv2.putText(
                frame, 
                "Test Video", 
                (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.5, 
                (0, 0, 0), 
                2
            )
            
            writer.write(frame)
    finally:
        writer.release()
    
    return video_path


@pytest.fixture
def nonexistent_video() -> Path:
    """Path to a video that doesn't exist."""
    return Path("/nonexistent/path/to/video.mp4")


@pytest.fixture
def invalid_video(temp_dir: Path) -> Path:
    """Create an invalid video file (not a real video)."""
    invalid_path = temp_dir / "invalid.mp4"
    invalid_path.write_text("This is not a video file")
    return invalid_path
