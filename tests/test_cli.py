"""Tests for command-line interface."""

import json
from pathlib import Path

import pytest

from video_feature_extractor.cli import main, create_parser


class TestCLI:
    """Tests for CLI functionality."""
    
    def test_parser_creation(self):
        """Should create argument parser."""
        parser = create_parser()
        assert parser is not None
    
    def test_version_flag(self, capsys):
        """Should print version with --version flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "2.0.0" in captured.out
    
    def test_help_flag(self, capsys):
        """Should print help with --help flag."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Extract visual and temporal features" in captured.out
    
    def test_check_availability(self, capsys):
        """Should check feature availability."""
        # Need a dummy video path even though we're just checking
        result = main(["dummy.mp4", "--check-availability"])
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Feature Availability" in captured.out
    
    def test_missing_video(self, capsys):
        """Should return error for missing video."""
        result = main(["/nonexistent/video.mp4", "--features", "cuts"])
        
        assert result != 0
        captured = capsys.readouterr()
        assert "Error" in captured.err or "not found" in captured.err.lower()
    
    def test_extract_to_file(self, generated_video: Path, temp_dir: Path, capsys):
        """Should extract features and save to file."""
        output_path = temp_dir / "output.json"
        
        result = main([
            str(generated_video),
            "--features", "cuts",
            "--output", str(output_path),
            "--quiet"
        ])
        
        assert result == 0
        assert output_path.is_file()
        
        # Verify JSON content
        data = json.loads(output_path.read_text())
        assert "results" in data
    
    def test_verbose_mode(self, generated_video: Path):
        """Should accept verbose flag."""
        result = main([
            str(generated_video),
            "--features", "cuts",
            "-v"
        ])
        
        assert result == 0
    
    def test_compact_output(self, generated_video: Path, capsys):
        """Should output compact JSON with --compact."""
        result = main([
            str(generated_video),
            "--features", "cuts",
            "--compact"
        ])
        
        assert result == 0
        captured = capsys.readouterr()
        # Compact JSON should not have newlines in the result dict
        assert captured.out.count('\n') <= 2  # Just the final newline
