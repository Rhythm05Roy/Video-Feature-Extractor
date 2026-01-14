# Video Feature Extraction Tool

Enterprise-grade Python tool for analyzing video files and extracting visual/temporal features. Built with OpenCV, pytesseract, and YOLOv8.

## Features

| Feature | Description | Technology |
|---------|-------------|------------|
| **Shot Cut Detection** | Counts hard cuts using frame-to-frame pixel analysis | OpenCV |
| **Motion Analysis** | Computes average motion magnitude via optical flow | OpenCV Farneback |
| **Text Detection (OCR)** | Detects text presence and extracts keywords | pytesseract |
| **Object/Person Detection** | Estimates person vs object dominance ratio | YOLOv8 (ultralytics) |

## Installation

### Prerequisites

- Python 3.9+
- Tesseract OCR (for text detection)
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: [Download installer](https://github.com/tesseract-ocr/tesseract)

### Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd video-feature-extractor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e ".[dev]"
# Or
pip install -r requirements.txt -r requirements-dev.txt
```

## Usage

### Command Line Interface

```bash
# Analyze with all features
video-feature-extractor video.mp4

# Select specific features
video-feature-extractor video.mp4 --features cuts motion

# Use configuration file
video-feature-extractor video.mp4 --config config.yaml

# Save output to file
video-feature-extractor video.mp4 --output results.json

# Verbose mode
video-feature-extractor video.mp4 -v

# Check feature availability
video-feature-extractor video.mp4 --check-availability

# View all options
video-feature-extractor --help
```

### Python API

```python
from video_feature_extractor import VideoFeatureExtractor, ExtractorConfig

# Default configuration
extractor = VideoFeatureExtractor()
results = extractor.extract("video.mp4", features=["cuts", "motion"])

# Custom configuration
config = ExtractorConfig.from_yaml("config.yaml")
extractor = VideoFeatureExtractor(config)
results = extractor.extract("video.mp4")

# Export to JSON
json_output = extractor.extract_to_json("video.mp4", output_path="results.json")

# Check availability
availability = extractor.check_availability()
print(availability)  # {'cuts': True, 'motion': True, 'text': True, 'objects': True}
```

### Configuration

Create a `config.yaml` file (see `config.example.yaml`):

```yaml
shot_cut:
  frame_step: 1
  diff_threshold: 30.0
  min_gap_frames: 5

motion:
  frame_step: 2

text_detection:
  frame_step: 15
  min_confidence: 70.0

object_detection:
  frame_step: 15
  confidence_threshold: 0.5
  model_size: "n"  # n, s, m, l, x
  use_gpu: false

logging:
  level: "INFO"
```

Environment variable overrides:

```bash
export VFE_LOGGING_LEVEL=DEBUG
export VFE_OBJECT_DETECTION_USE_GPU=true
```

## Output Format

```json
{
  "video_path": "/videos/sample.mp4",
  "video_metadata": {
    "resolution": {"width": 1920, "height": 1080},
    "fps": 30.0,
    "duration_seconds": 120.5,
    "codec": "h264",
    "total_frames": 3615
  },
  "extraction_timestamp": "2024-01-14T14:30:00Z",
  "features_requested": ["cuts", "motion", "text", "objects"],
  "results": {
    "shot_cut_detection": {
      "shot_cut_count": 12,
      "cut_frames": [120, 450, 890],
      "frame_step_used": 1,
      "mean_diff_threshold": 30.0
    },
    "motion_analysis": {
      "average_motion_magnitude": 0.8421,
      "max_motion_magnitude": 2.345,
      "min_motion_magnitude": 0.012,
      "motion_samples": 240
    },
    "text_detection": {
      "text_present_ratio": 0.18,
      "frames_with_text": 9,
      "total_frames_evaluated": 50,
      "keywords_top10": ["intro", "title", "sample"]
    },
    "object_person_dominance": {
      "persons_detected": 42,
      "objects_detected": 61,
      "person_ratio": 0.41,
      "object_ratio": 0.59,
      "dominant_category": "object",
      "class_distribution": {"person": 42, "car": 30, "chair": 15}
    }
  },
  "processing_time_seconds": 45.2,
  "extractor_version": "2.0.0"
}
```

## Project Structure

```
video-feature-extractor/
├── src/
│   └── video_feature_extractor/
│       ├── __init__.py          # Package exports
│       ├── core.py              # Main orchestrator
│       ├── cli.py               # Command-line interface
│       ├── config.py            # Configuration management
│       ├── exceptions.py        # Custom exceptions
│       ├── logging_config.py    # Logging setup
│       ├── extractors/          # Feature extractors
│       │   ├── base.py          # Abstract base class
│       │   ├── shot_cuts.py     # Shot cut detection
│       │   ├── motion.py        # Motion analysis
│       │   ├── text_ocr.py      # Text/OCR detection
│       │   └── object_detection.py  # YOLO detection
│       └── utils/               # Utilities
│           └── video.py         # Video helpers
├── tests/                       # Test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_extractors.py       # Unit tests
│   ├── test_core.py             # Integration tests
│   └── test_cli.py              # CLI tests
├── config.example.yaml          # Example configuration
├── pyproject.toml               # Build configuration
├── requirements.txt             # Dependencies
└── requirements-dev.txt         # Dev dependencies
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/video_feature_extractor --cov-report=html

# Run specific test file
pytest tests/test_extractors.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/video_feature_extractor

# Linting
flake8 src/ tests/
```

## Legacy Support

The original single-file implementation is preserved at `src/video_feature_extractor.py` for backward compatibility:

```bash
python -m src.video_feature_extractor video.mp4
```

## License

MIT License
