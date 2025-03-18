#!/usr/bin/env python3
"""
Script to set up git hooks for the project.
This script creates a pre-commit hook that runs Black on the src directory.
"""

import sys
import os
from pathlib import Path
import stat

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Git hooks directory
HOOKS_DIR = project_root / ".git" / "hooks"


def create_pre_commit_hook():
    """Create a pre-commit hook that runs Black on the src directory."""
    pre_commit_path = HOOKS_DIR / "pre-commit"

    # Create hooks directory if it doesn't exist
    HOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-commit hook content
    pre_commit_content = """#!/bin/sh
# Pre-commit hook to run Black on the src directory

echo "Running Black on src directory..."
python -m black src/ || {
    echo "Black failed! Fix the issues before committing."
    exit 1
}

echo "Black passed successfully!"
"""

    # Write the pre-commit hook
    with open(pre_commit_path, "w") as f:
        f.write(pre_commit_content)

    # Make the hook executable
    pre_commit_path.chmod(pre_commit_path.stat().st_mode | stat.S_IEXEC)

    print(f"Pre-commit hook created at {pre_commit_path}")
    print("The hook will run Black on the src directory before each commit.")


def main():
    print("Setting up git hooks...")

    if not (project_root / ".git").exists():
        print("Error: .git directory not found. Make sure you're in a git repository.")
        sys.exit(1)

    create_pre_commit_hook()
    print("Git hooks setup complete!")


if __name__ == "__main__":
    main()
