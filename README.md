<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1b2a,50:1b263b,100:415a77&height=180&section=header&text=Video%20Feature%20Extractor&fontSize=46&fontColor=ffffff&fontAlignY=38&desc=Shot%20Cuts%20%C2%B7%20Motion%20%C2%B7%20OCR%20%C2%B7%20Object%20Detection%20Pipeline&descAlignY=58&descSize=18&descColor=9db4c0" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-43A047?style=flat-square)
![Pytest](https://img.shields.io/badge/Tested%20with-Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

**An enterprise-grade Python toolkit that analyzes video files and extracts structured visual and temporal features — shot cuts, motion intensity, on-screen text, and object/person dominance — via a unified config-driven pipeline.**

</div>

---

## Overview

Video Feature Extractor is a modular analysis pipeline that turns raw video files into structured, machine-readable feature reports. Each analysis dimension — shot cut detection, motion analysis, OCR-based text detection, and YOLO-based object/person detection — is implemented as an independently testable extractor behind a common abstract interface, coordinated by a single orchestrating facade class.

The pipeline is driven by **YAML-based configuration** (with environment-variable overrides), exposes both a **CLI** and a **Python API**, and emits a single structured **JSON report** per video containing metadata, per-feature results, and processing diagnostics.

---

## Pipeline Architecture

```
                    Video File (.mp4, .avi, .mov, ...)
                              │
                              ▼
                  ┌────────────────────────┐
                  │   Video Metadata        │  ── Resolution, FPS, duration,
                  │   Validation            │     codec, total frame count
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  VideoFeatureExtractor  │  ── Orchestrator facade,
                  │  (core.py)              │     resolves config + features
                  └────────────┬────────────┘
                               │
        ┌─────────────┬────────┴────────┬─────────────────┐
        ▼             ▼                 ▼                 ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐
│ ShotCut       │ │ Motion        │ │ TextOCR       │ │ ObjectDetection    │
│ Extractor     │ │ Extractor     │ │ Extractor     │ │ Extractor          │
│ (OpenCV       │ │ (OpenCV       │ │ (pytesseract) │ │ (YOLOv8 /          │
│  frame diff)  │ │  Farneback    │ │               │ │  ultralytics)      │
│               │ │  optical flow)│ │               │ │                    │
└──────┬────────┘ └──────┬────────┘ └──────┬────────┘ └─────────┬──────────┘
       │                 │                 │                    │
       └─────────────────┴─────────────────┴────────────────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │   JSON Result Builder   │  ── Merges metadata + per-feature
                  │                         │     results + timing info
                  └────────────┬────────────┘
                               │
                               ▼
                     results.json / stdout
```

---

## Features

| Feature | Description | Technology |
|---|---|---|
| **Shot Cut Detection** | Counts hard cuts via frame-to-frame pixel-difference analysis against a configurable threshold | OpenCV |
| **Motion Analysis** | Computes average / min / max motion magnitude via dense optical flow | OpenCV (Farneback) |
| **Text Detection (OCR)** | Detects on-screen text presence and extracts top keywords across sampled frames | pytesseract (Tesseract OCR) |
| **Object/Person Detection** | Detects objects and people, computes a person-vs-object dominance ratio and class distribution | YOLOv8 (Ultralytics) |

Each extractor:
- Implements a shared `BaseExtractor` abstract interface (`extract()`, `is_available()`)
- Supports a configurable **frame step** to trade off accuracy vs. processing time
- Reports progress via an optional callback, with debug-level logging
- Declares its own availability check, so the pipeline can gracefully report which features are runnable in the current environment

---

## Project Structure

```text
Video-Feature-Extractor/
├── src/
│   ├── video_feature_extractor/
│   │   ├── __init__.py              # Package exports & version
│   │   ├── core.py                  # VideoFeatureExtractor — main orchestrator facade
│   │   ├── cli.py                   # Command-line interface
│   │   ├── config.py                # ExtractorConfig — YAML + env var configuration
│   │   ├── exceptions.py            # Custom exception hierarchy
│   │   ├── logging_config.py        # Logging setup + ProgressLogger
│   │   ├── extractors/
│   │   │   ├── __init__.py          # EXTRACTOR_MAP registry
│   │   │   ├── base.py              # BaseExtractor abstract class
│   │   │   ├── shot_cuts.py         # Shot cut detection extractor
│   │   │   ├── motion.py            # Optical-flow motion extractor
│   │   │   ├── text_ocr.py          # Tesseract OCR text extractor
│   │   │   └── object_detection.py  # YOLOv8 object/person extractor
│   │   └── utils/
│   │       └── video.py             # Video metadata & validation helpers
│   └── video_feature_extractor.py   # Legacy single-file implementation
├── tests/
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_extractors.py           # Per-extractor unit tests
│   ├── test_core.py                 # Orchestrator integration tests
│   └── test_cli.py                  # CLI argument & I/O tests
├── config.example.yaml              # Example configuration file
├── pyproject.toml                   # Build configuration & package metadata
├── requirements.txt                 # Runtime dependencies
└── requirements-dev.txt             # Development dependencies
```

---

## Component Details

### `core.py` — `VideoFeatureExtractor`

The main facade class. Coordinates extractor selection, runs each requested extractor against the input video, merges results with video metadata, and produces the final structured report — either as a Python dict or serialized JSON.

```python
extractor = VideoFeatureExtractor()
results = extractor.extract("video.mp4", features=["cuts", "motion"])
```

It also exposes `check_availability()`, which probes each registered extractor's `is_available()` to report which features can actually run given the current environment (e.g. whether Tesseract or a YOLO model file is present).

### `extractors/` — Pluggable feature extractors

| Extractor | Feature key | Key config options |
|---|---|---|
| `ShotCutExtractor` | `cuts` | `frame_step`, `diff_threshold`, `min_gap_frames` |
| `MotionExtractor` | `motion` | `frame_step` |
| `TextOCRExtractor` | `text` | `frame_step`, `min_confidence` |
| `ObjectDetectionExtractor` | `objects` | `frame_step`, `confidence_threshold`, `model_size`, `use_gpu` |

All extractors inherit from `BaseExtractor` (`extractors/base.py`), which defines the abstract `extract()` / `is_available()` contract, a shared `on_progress()` reporting hook, and config-validation extension points.

### `config.py` — `ExtractorConfig`

Loads configuration from a YAML file (`ExtractorConfig.from_yaml(...)`) with sensible defaults, and supports environment-variable overrides (e.g. `VFE_LOGGING_LEVEL`, `VFE_OBJECT_DETECTION_USE_GPU`) for containerized or CI environments.

### `cli.py` — Command-line interface

Wraps `VideoFeatureExtractor` with an argument-parsed CLI supporting feature selection, config file paths, JSON output, verbosity flags, and an availability-check mode.

### `utils/video.py`

Shared helpers for video file validation and metadata extraction (resolution, FPS, duration, codec, frame count) used by both the core orchestrator and individual extractors.

### `exceptions.py` — Exception hierarchy

A dedicated exception hierarchy (`VideoFeatureExtractorError`, `VideoNotFoundError`, `VideoOpenError`, `ModelNotFoundError`, `OCRError`, `InvalidFeatureError`) gives callers fine-grained control over error handling without parsing string messages.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Tesseract OCR (required for text detection)
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: [Download installer](https://github.com/tesseract-ocr/tesseract)

### Installation

```bash
git clone https://github.com/Rhythm05Roy/Video-Feature-Extractor.git
cd Video-Feature-Extractor

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package
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

---

## Usage

### Command Line Interface

```bash
# Analyze with all features
video-feature-extractor video.mp4

# Select specific features
video-feature-extractor video.mp4 --features cuts motion

# Use a configuration file
video-feature-extractor video.mp4 --config config.yaml

# Save output to file
video-feature-extractor video.mp4 --output results.json

# Verbose mode
video-feature-extractor video.mp4 -v

# Check which features are runnable in this environment
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

# Export directly to JSON
json_output = extractor.extract_to_json("video.mp4", output_path="results.json")

# Check availability
availability = extractor.check_availability()
print(availability)  # {'cuts': True, 'motion': True, 'text': True, 'objects': True}
```

---

## Configuration

Create a `config.yaml` (see `config.example.yaml`):

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

### Environment variable overrides

```bash
export VFE_LOGGING_LEVEL=DEBUG
export VFE_OBJECT_DETECTION_USE_GPU=true
```

---

## Output Format

Every run produces a single structured JSON report:

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

| Section | Description |
|---|---|
| `video_metadata` | Resolution, FPS, duration, codec, and total frame count |
| `features_requested` | Which feature keys were requested for this run |
| `results.shot_cut_detection` | Cut count, cut frame indices, frame step, and threshold used |
| `results.motion_analysis` | Average / max / min optical-flow motion magnitude and sample count |
| `results.text_detection` | Text presence ratio, frames evaluated, and top OCR keywords |
| `results.object_person_dominance` | Detection counts, person/object ratios, dominant category, and class distribution |
| `processing_time_seconds` | Wall-clock time for the full extraction run |

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/video_feature_extractor --cov-report=html

# Run a specific test file
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

---

## Key Design Patterns

- **Abstract extractor interface** — every feature extractor implements `BaseExtractor.extract()` and `is_available()`, so new features (e.g. scene classification, audio analysis) can be added without touching the orchestrator
- **Config-driven behavior** — all thresholds, frame steps, and model sizes live in YAML, with environment-variable overrides for deployment flexibility
- **Graceful availability checks** — `check_availability()` lets callers detect missing dependencies (Tesseract, YOLO weights) before committing to a full extraction run
- **Structured exception hierarchy** — dedicated exception types (`VideoNotFoundError`, `OCRError`, `ModelNotFoundError`, etc.) enable precise error handling in calling code
- **Dual interface** — the same `VideoFeatureExtractor` core powers both the CLI (`video-feature-extractor`) and the importable Python API
- **Backward-compatible legacy mode** — the original single-file implementation remains runnable via `python -m src.video_feature_extractor video.mp4`

---

## Legacy Support

The original single-file implementation is preserved at `src/video_feature_extractor.py` for backward compatibility:

```bash
python -m src.video_feature_extractor video.mp4
```

---

## License

MIT — see `LICENSE`.

<div align="center">

**Built by [Ridam Roy](https://github.com/Rhythm05Roy)**

[![GitHub](https://img.shields.io/badge/GitHub-Rhythm05Roy-181717?style=flat-square&logo=github)](https://github.com/Rhythm05Roy)
[![Email](https://img.shields.io/badge/Email-ridam15--4260%40diu.edu.bd-D44638?style=flat-square&logo=gmail&logoColor=white)](mailto:ridam15-4260@diu.edu.bd)

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1b2a,50:1b263b,100:415a77&height=80&section=footer" width="100%"/>

</div>
