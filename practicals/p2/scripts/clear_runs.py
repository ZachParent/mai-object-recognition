#!/usr/bin/env python3
"""
Script to clear the runs directory (data/01_runs).
This script deletes all files and subdirectories in the runs directory
to clean up after previous training runs.
"""

import sys
import shutil
from pathlib import Path
import argparse

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.config import RUNS_DIR, METRICS_DIR


def clear_directory(directory: Path, dry_run: bool = False) -> None:
    """
    Clear all contents of a directory without deleting the directory itself.

    Args:
        directory: Path to the directory to clear
        dry_run: If True, only print what would be deleted without actually deleting
    """
    if not directory.exists():
        print(f"Directory {directory} does not exist. Nothing to clear.")
        return

    # Count files and directories to be deleted
    files_count = 0
    dirs_count = 0

    for item in directory.iterdir():
        if item.is_file():
            files_count += 1
            print(f"{'Would delete' if dry_run else 'Deleting'} file: {item}")
            if not dry_run:
                item.unlink()
        elif item.is_dir():
            dirs_count += 1
            print(f"{'Would delete' if dry_run else 'Deleting'} directory: {item}")
            if not dry_run:
                shutil.rmtree(item)

    print(
        f"{'Would have deleted' if dry_run else 'Deleted'} {files_count} files and {dirs_count} directories from {directory}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clear the runs and/or metrics directories."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--metrics", action="store_true", help="Also clear the metrics directory"
    )
    args = parser.parse_args()

    print(f"{'DRY RUN: ' if args.dry_run else ''}Clearing runs directory...")
    clear_directory(RUNS_DIR, args.dry_run)

    if args.metrics:
        print(f"{'DRY RUN: ' if args.dry_run else ''}Clearing metrics directory...")
        clear_directory(METRICS_DIR, args.dry_run)

    print("Clearing complete!")


if __name__ == "__main__":
    main()
