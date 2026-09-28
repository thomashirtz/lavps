import sys
import json
from pathlib import Path
from unittest.mock import patch

from lavps.paths import OUTPUT_DIRECTORY, REPOSITORY_PATH

TARGET_FOLDER = OUTPUT_DIRECTORY / "compute_metrics"
SCRIPT_PATH = REPOSITORY_PATH / "scripts" / "compute_metrics.py"

def test_compute_metrics():
    parent_folder = Path(TARGET_FOLDER)
    output_file = parent_folder.with_name(parent_folder.name + ".jsonl")

    test_args = [
        "calculate_metrics.py",
        "-p", str(parent_folder),
        "-o", str(output_file),
        "--fid_n", "1",
    ]

    try:
        with patch.object(sys, 'argv', test_args):
            import runpy
            try:
                runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
            except SystemExit:
                pass

        assert output_file.exists(), f"Expected output file {output_file} was not created."
        with open(output_file, 'r') as f:
            results = [json.loads(line) for line in f]
        print(f"\nSuccessfully processed {len(results)} folders.")

    finally:
        if output_file.exists():
            output_file.unlink()
            print(f"\nCleaned up: {output_file}")
