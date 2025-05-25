# Preprocessing Script for CLOTH3D++ Dataset

This script processes the CLOTH3D++ dataset to prepare data for depth estimation training. It performs the following operations:

- Extracts and merges video frames from RGB and segmentation videos
- Renders and crops depth maps with consistent center and margins
- Creates SMPL skeleton pose visualizations
- Processes RGB frames with transparent backgrounds
- Resizes all outputs to 256x256 pixels

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- PIL (Python Imaging Library)
- ffmpeg (for video frame extraction)

## Data Setup

Before running the script, make sure your data is properly set up:

1. **SMPL Models**: Place the SMPL model files in the directory specified by `SMPL_DIR` in `config.py`
2. **Raw Dataset**: Place the raw CLOTH3D++ dataset files in the directory specified by `RAW_DATA_DIR` in `config.py`

These paths can be modified in the `config.py` file to match your local setup if needed.

## Usage

**Important:** Run this script from the project's root directory, not from within the scripts folder.

```bash
python -m practicals.p3.scripts.unified_preprocessing.preprocessing [options]
```

### Options

- `--no-depth`: Skip depth map processing and saving
- `--no-pose`: Skip SMPL pose visualization processing and saving

## Output Structure

The script creates the following directory structure for each sample:

```
preprocessed_dataset/
  └── sample_name/
      ├── depth/         # NPY depth maps
      ├── depth_vis/     # Visualization images of depth maps
      ├── rgb/           # Processed RGB frames
      └── pose/          # SMPL skeleton visualizations
```

## Example

```bash
# Process all data types
python -m scripts.unified_preprocessing.preprocessing

# Process only RGB and pose data
python -m scripts.unified_preprocessing.preprocessing --no-depth

# Process only RGB and depth data
python -m scripts.unified_preprocessing.preprocessing --no-pose
```
