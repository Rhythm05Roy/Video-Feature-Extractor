"""Text detection (OCR) extractor using pytesseract.

Detects and extracts text from video frames using Tesseract OCR.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np

from video_feature_extractor.extractors.base import BaseExtractor, ProgressCallback
from video_feature_extractor.config import ExtractorConfig, TextDetectionConfig
from video_feature_extractor.exceptions import VideoOpenError, OCRError

# Try to import pytesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None


class TextOCRExtractor(BaseExtractor):
    """Extractor for detecting text in video frames using OCR.
    
    Samples frames from the video and uses Tesseract OCR to detect
    and extract text, calculating presence ratio and common keywords.
    
    Example:
        >>> config = ExtractorConfig()
        >>> extractor = TextOCRExtractor(config)
        >>> results = extractor.extract(Path("video.mp4"))
        >>> print(results["text_present_ratio"])
        0.18
    """
    
    name = "Text Detection (OCR)"
    feature_key = "text_detection"
    
    def __init__(
        self,
        config: ExtractorConfig,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[ProgressCallback] = None
    ):
        super().__init__(config, logger, progress_callback)
        self._config = config.text_detection
        self._tesseract_checked = False
        self._tesseract_installed = False
    
    def get_config_section(self) -> TextDetectionConfig:
        return self._config
    
    def is_available(self) -> bool:
        """Check if Tesseract OCR is installed and available."""
        if not TESSERACT_AVAILABLE:
            return False
        
        if not self._tesseract_checked:
            try:
                pytesseract.get_tesseract_version()
                self._tesseract_installed = True
            except pytesseract.TesseractNotFoundError:
                self._tesseract_installed = False
            self._tesseract_checked = True
        
        return self._tesseract_installed
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better OCR results.
        
        Args:
            frame: BGR frame from video.
            
        Returns:
            Preprocessed binary image.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    def _extract_text_from_frame(
        self, 
        frame: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """Extract text from a single frame.
        
        Args:
            frame: BGR frame from video.
            
        Returns:
            Tuple of (has_text, list_of_words).
        """
        preprocessed = self._preprocess_frame(frame)
        min_conf = self._config.min_confidence
        
        try:
            data = pytesseract.image_to_data(
                preprocessed, 
                output_type=pytesseract.Output.DICT,
                lang=self._config.language
            )
        except pytesseract.TesseractNotFoundError:
            raise OCRError(
                "Tesseract OCR binary not found. Install it and add to PATH.",
                tesseract_installed=False
            )
        
        has_text = False
        keywords: List[str] = []
        
        for word, conf in zip(data.get("text", []), data.get("conf", [])):
            if not word or word.isspace():
                continue
            try:
                conf_val = float(conf)
            except (ValueError, TypeError):
                continue
            
            if conf_val >= min_conf:
                has_text = True
                cleaned = word.strip()
                if len(cleaned) >= 2:  # Filter very short words
                    keywords.append(cleaned)
        
        return has_text, keywords
    
    def extract(self, video_path: Path) -> Dict[str, Any]:
        """Detect text presence in video frames.
        
        Args:
            video_path: Path to the video file.
            
        Returns:
            Dictionary with:
                - text_present_ratio: Ratio of frames containing text
                - frames_with_text: Count of frames with detected text
                - total_frames_evaluated: Total frames processed
                - keywords_top10: Most common words detected
                - unique_words: Count of unique words found
                - frame_step_used: Actual frame step used
                - min_confidence: OCR confidence threshold used
                
        Raises:
            VideoOpenError: If the video cannot be opened.
            OCRError: If Tesseract is not available.
        """
        if not self.is_available():
            self.logger.warning("Tesseract OCR not available, skipping text detection")
            return {
                "text_present_ratio": 0.0,
                "frames_with_text": 0,
                "total_frames_evaluated": 0,
                "keywords_top10": [],
                "error": "Tesseract OCR not installed or not in PATH",
                "available": False,
            }
        
        self.logger.info(f"Starting text detection on: {video_path}")
        
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
            
            frames_evaluated = 0
            frames_with_text = 0
            keywords: Counter = Counter()
            
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                
                frames_evaluated += 1
                
                try:
                    has_text, words = self._extract_text_from_frame(frame)
                except OCRError as e:
                    self.logger.error(str(e))
                    return {
                        "text_present_ratio": 0.0,
                        "frames_with_text": 0,
                        "total_frames_evaluated": frames_evaluated,
                        "keywords_top10": [],
                        "error": str(e),
                        "available": False,
                    }
                
                if has_text:
                    frames_with_text += 1
                    keywords.update([w.lower() for w in words])
                
                # Report progress
                if frames_evaluated % 10 == 0:
                    self.on_progress(
                        frames_evaluated, 
                        estimated_samples, 
                        f"{frames_with_text} with text"
                    )
                
                # Frame stepping
                if cfg.frame_step > 1:
                    for _ in range(cfg.frame_step - 1):
                        capture.grab()
            
            self.on_progress(estimated_samples, estimated_samples, "Complete")
            
        finally:
            capture.release()
        
        ratio = frames_with_text / frames_evaluated if frames_evaluated > 0 else 0.0
        most_common = [word for word, _ in keywords.most_common(10)]
        
        self.logger.info(
            f"Text detection complete: {frames_with_text}/{frames_evaluated} frames "
            f"have text ({ratio:.1%}), {len(keywords)} unique words"
        )
        
        return {
            "text_present_ratio": round(ratio, 4),
            "frames_with_text": frames_with_text,
            "total_frames_evaluated": frames_evaluated,
            "keywords_top10": most_common,
            "unique_words": len(keywords),
            "frame_step_used": cfg.frame_step,
            "min_confidence": cfg.min_confidence,
            "language": cfg.language,
            "available": True,
        }
