"""Video utility functions for Video Feature Extractor.

Provides common video operations including:
- Video file validation
- Metadata extraction
- Efficient frame iteration
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple
import cv2
import numpy as np

from video_feature_extractor.exceptions import VideoNotFoundError, VideoOpenError


@dataclass
class VideoMetadata:
    """Container for video metadata.
    
    Attributes:
        path: Absolute path to the video file.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        total_frames: Total number of frames.
        duration_seconds: Duration in seconds.
        codec: FourCC codec identifier.
        file_size_bytes: Size of the video file in bytes.
    """
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    file_size_bytes: int
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            "path": self.path,
            "resolution": {"width": self.width, "height": self.height},
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "codec": self.codec,
            "file_size_bytes": self.file_size_bytes
        }


def validate_video_file(video_path: Path) -> Path:
    """Validate that a video file exists and is accessible.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        Resolved absolute path to the video.
        
    Raises:
        VideoNotFoundError: If the file does not exist.
    """
    path = Path(video_path).resolve()
    if not path.is_file():
        raise VideoNotFoundError(str(video_path))
    return path


def get_video_metadata(video_path: Path) -> VideoMetadata:
    """Extract comprehensive metadata from a video file.
    
    Args:
        video_path: Path to the video file.
        
    Returns:
        VideoMetadata instance with all extracted information.
        
    Raises:
        VideoNotFoundError: If the file does not exist.
        VideoOpenError: If the video cannot be opened.
    """
    path = validate_video_file(video_path)
    
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoOpenError(str(path), "OpenCV could not open the video file")
    
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get codec as FourCC string
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0.0
        
        # Get file size
        file_size = path.stat().st_size
        
        return VideoMetadata(
            path=str(path),
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec.strip(),
            file_size_bytes=file_size
        )
    finally:
        capture.release()


class FrameIterator:
    """Efficient frame iterator with stepping support.
    
    This iterator allows processing video frames with configurable stepping,
    reducing computational load for analysis that doesn't require every frame.
    
    Example:
        with FrameIterator(video_path, frame_step=5) as frames:
            for frame_idx, frame in frames:
                # Process every 5th frame
                pass
    
    Attributes:
        video_path: Path to the video file.
        frame_step: Process every Nth frame.
        start_frame: Frame index to start from.
        end_frame: Frame index to end at (None for end of video).
        convert_gray: If True, yield grayscale frames.
    """
    
    def __init__(
        self,
        video_path: Path,
        frame_step: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        convert_gray: bool = False
    ):
        self.video_path = Path(video_path)
        self.frame_step = max(1, frame_step)
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.convert_gray = convert_gray
        self._capture: Optional[cv2.VideoCapture] = None
        self._current_frame = 0
        self._total_frames = 0
    
    def __enter__(self) -> "FrameIterator":
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            raise VideoOpenError(str(self.video_path))
        
        self._total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Seek to start frame if specified
        if self.start_frame > 0:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
            self._current_frame = self.start_frame
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
    
    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        return self
    
    def __next__(self) -> Tuple[int, np.ndarray]:
        if self._capture is None:
            raise RuntimeError("FrameIterator must be used as context manager")
        
        # Check end condition
        if self.end_frame is not None and self._current_frame >= self.end_frame:
            raise StopIteration
        
        ok, frame = self._capture.read()
        if not ok:
            raise StopIteration
        
        current_idx = self._current_frame
        
        # Skip frames for stepping
        if self.frame_step > 1:
            for _ in range(self.frame_step - 1):
                self._capture.grab()
                self._current_frame += 1
        
        self._current_frame += 1
        
        if self.convert_gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return current_idx, frame
    
    @property
    def total_frames(self) -> int:
        """Total number of frames in the video."""
        return self._total_frames
    
    @property
    def processed_frames(self) -> int:
        """Estimated number of frames that will be processed with current settings."""
        end = self.end_frame or self._total_frames
        start = self.start_frame
        return max(0, (end - start + self.frame_step - 1) // self.frame_step)


def get_frame_at_position(video_path: Path, frame_number: int) -> Optional[np.ndarray]:
    """Get a specific frame from a video.
    
    Args:
        video_path: Path to the video file.
        frame_number: Frame index (0-based).
        
    Returns:
        Frame as numpy array, or None if frame couldn't be read.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()
