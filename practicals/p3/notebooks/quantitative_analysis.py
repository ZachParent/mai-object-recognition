import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.config import CHECKPOINTS_DIR, RESULTS_DIR
from src.datasets.cloth3d import Cloth3dDataset
from src.inferrer import load_model
from src.metrics import compute_metrics

# Load the trained model
model_path = str(CHECKPOINTS_DIR / "run_0.pt")
inferrer = load_model(model_path)

# Create dataset
dataset = Cloth3dDataset(start_idx=0, enable_normalization=True)

# Initialize lists to store results
results = []

# Process each video
for video_id in tqdm(dataset.video_ids, desc="Processing videos"):
    # Get all frames for this video
    video_frames = [
        i for i in range(len(dataset)) if dataset.get_video_id(i) == video_id
    ]

    # Process each frame
    for frame_idx in tqdm(
        video_frames, desc=f"Processing frames for video {video_id}", leave=False
    ):
        # Get data
        image, depth = dataset[frame_idx]

        # Get prediction
        with torch.no_grad():
            pred_depth = inferrer.model(image.unsqueeze(0))
            pred_depth = pred_depth.squeeze(0)

        # Compute metrics
        metrics = compute_metrics(pred_depth, depth)

        # Store results
        results.append(
            {
                "video_id": video_id,
                "frame_id": dataset.get_frame_id(frame_idx),
                **metrics,
            }
        )

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
results_df.to_csv(RESULTS_DIR / "quantitative_analysis.csv", index=False)

# Print summary statistics
print("\nSummary Statistics:")
print(results_df.describe())

# Group by video and compute mean metrics
video_metrics = results_df.groupby("video_id").mean()
print("\nMean Metrics by Video:")
print(video_metrics)

# Save video-level metrics
video_metrics.to_csv(RESULTS_DIR / "video_metrics.csv")
