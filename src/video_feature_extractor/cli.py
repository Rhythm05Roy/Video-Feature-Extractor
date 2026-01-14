"""Command-line interface for Video Feature Extractor.

Provides a comprehensive CLI with support for configuration files,
verbose output, and multiple output formats.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from video_feature_extractor import __version__
from video_feature_extractor.config import ExtractorConfig
from video_feature_extractor.core import VideoFeatureExtractor
from video_feature_extractor.exceptions import VideoFeatureExtractorError


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="video-feature-extractor",
        description="Extract visual and temporal features from video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with default settings
  video-feature-extractor video.mp4

  # Select specific features
  video-feature-extractor video.mp4 --features cuts motion

  # Use a configuration file
  video-feature-extractor video.mp4 --config config.yaml

  # Save output to file
  video-feature-extractor video.mp4 --output results.json

  # Verbose output
  video-feature-extractor video.mp4 -v
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "video_path",
        type=Path,
        help="Path to the video file to analyze."
    )
    
    # Feature selection
    parser.add_argument(
        "-f", "--features",
        nargs="+",
        choices=["cuts", "motion", "text", "objects"],
        default=None,
        help="Features to extract. Default: all available features."
    )
    
    # Configuration
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to YAML/JSON configuration file."
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Path to save JSON output. Default: print to stdout."
    )
    
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Output compact JSON (no indentation)."
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude video metadata from output."
    )
    
    # Logging options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (DEBUG level)."
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all output except errors and results."
    )
    
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file."
    )
    
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="Output logs in JSON format."
    )
    
    # Shot cut options
    parser.add_argument(
        "--cut-frame-step",
        type=int,
        help="Frame step for shot cut detection."
    )
    
    parser.add_argument(
        "--diff-threshold",
        type=float,
        help="Mean pixel difference threshold for hard cuts."
    )
    
    parser.add_argument(
        "--min-gap-frames",
        type=int,
        help="Minimum frame gap between detected hard cuts."
    )
    
    # Motion options
    parser.add_argument(
        "--motion-frame-step",
        type=int,
        help="Frame step for motion analysis."
    )
    
    # Text detection options
    parser.add_argument(
        "--text-frame-step",
        type=int,
        help="Frame step for OCR sampling."
    )
    
    parser.add_argument(
        "--text-min-confidence",
        type=float,
        help="Minimum OCR confidence score."
    )
    
    # Object detection options
    parser.add_argument(
        "--object-frame-step",
        type=int,
        help="Frame step for object detection."
    )
    
    parser.add_argument(
        "--object-conf-threshold",
        type=float,
        help="YOLO confidence threshold."
    )
    
    parser.add_argument(
        "--yolo-model-size",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)."
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for object detection (if available)."
    )
    
    # Utility options
    parser.add_argument(
        "--check-availability",
        action="store_true",
        help="Check which features are available and exit."
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    return parser


def apply_cli_overrides(config: ExtractorConfig, args: argparse.Namespace) -> ExtractorConfig:
    """Apply CLI argument overrides to configuration.
    
    Args:
        config: Base configuration.
        args: Parsed CLI arguments.
        
    Returns:
        Configuration with CLI overrides applied.
    """
    # Logging
    if args.verbose:
        config.logging.level = "DEBUG"
    elif args.quiet:
        config.logging.level = "ERROR"
    
    if args.log_file:
        config.logging.log_file = str(args.log_file)
    
    if args.json_logs:
        config.logging.json_format = True
    
    # Shot cuts
    if args.cut_frame_step:
        config.shot_cut.frame_step = args.cut_frame_step
    if args.diff_threshold:
        config.shot_cut.diff_threshold = args.diff_threshold
    if args.min_gap_frames:
        config.shot_cut.min_gap_frames = args.min_gap_frames
    
    # Motion
    if args.motion_frame_step:
        config.motion.frame_step = args.motion_frame_step
    
    # Text detection
    if args.text_frame_step:
        config.text_detection.frame_step = args.text_frame_step
    if args.text_min_confidence:
        config.text_detection.min_confidence = args.text_min_confidence
    
    # Object detection
    if args.object_frame_step:
        config.object_detection.frame_step = args.object_frame_step
    if args.object_conf_threshold:
        config.object_detection.confidence_threshold = args.object_conf_threshold
    if args.yolo_model_size:
        config.object_detection.model_size = args.yolo_model_size
    if args.use_gpu:
        config.object_detection.use_gpu = True
    
    return config


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point.
    
    Args:
        argv: Command line arguments (uses sys.argv if None).
        
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    try:
        # Load configuration
        if args.config:
            config = ExtractorConfig.from_file(args.config)
        else:
            config = ExtractorConfig()
        
        # Apply CLI overrides
        config = apply_cli_overrides(config, args)
        
        # Apply environment variable overrides
        config.apply_env_overrides()
        
        # Create extractor
        extractor = VideoFeatureExtractor(config)
        
        # Check availability mode
        if args.check_availability:
            availability = extractor.check_availability()
            print("Feature Availability:")
            for feature, available in availability.items():
                status = "✓ Available" if available else "✗ Not available"
                print(f"  {feature}: {status}")
            return 0
        
        # Extract features
        results = extractor.extract(
            video_path=args.video_path,
            features=args.features,
            include_metadata=not args.no_metadata
        )
        
        # Format output
        indent = None if args.compact else 2
        json_output = json.dumps(results, indent=indent, default=str)
        
        # Save or print output
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json_output)
            if not args.quiet:
                print(f"Results saved to: {args.output}")
        else:
            print(json_output)
        
        return 0
        
    except VideoFeatureExtractorError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
