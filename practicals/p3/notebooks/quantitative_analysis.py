# %%
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

# %%
# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.config import CHECKPOINTS_DIR, RESULTS_DIR
from src.datasets.cloth3d import Cloth3dDataset
from src.inferrer import load_model
from src.metrics import get_metric_collection
from src.run_configs import ModelName, RunConfig

# %%
# Load the trained model
run_id = 0
model_path = str(CHECKPOINTS_DIR / f"run_{run_id:03d}.pt")
inferrer = load_model(model_path)

# Create a RunConfig for metrics (adjust as needed)
run_config = RunConfig(
    id=run_id,
    name="quant_eval",
    model_name=ModelName.UNET2D,
    learning_rate=0.001,
    batch_size=1,
    epochs=1,
    perceptual_loss="L2",
    perceptual_loss_weight=0.5,
)
video_metric_collection = get_metric_collection(run_config)
frame_metric_collection = get_metric_collection(run_config)

# %%
# Create dataset
dataset = Cloth3dDataset(start_idx=0, enable_normalization=True)

# %%
# Initialize lists to store results
results = []
frame_results = []  # <-- new list for per-frame metrics

# Process each video
for video_id in tqdm(dataset.video_ids, desc="Processing videos"):
    # Get all frames for this video
    video_frames = [
        i for i in range(len(dataset)) if dataset.get_video_id(i) == video_id
    ]

    # Reset metrics for each video if you want per-video metrics
    video_metric_collection.reset()

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

        # Update metric collection
        frame_metric_collection.reset()
        frame_metric_collection.update(pred_depth, depth)
        video_metric_collection.update(pred_depth, depth)

        # Store per-frame metrics
        metrics = frame_metric_collection.compute()
        frame_results.append(
            {
                "run_id": run_id,
                "video_id": video_id,
                "frame_id": dataset.get_frame_id(frame_idx),
                **metrics,
            }
        )

    # After all frames in video, store per-video metrics
    metrics = video_metric_collection.compute()
    results.append(
        {
            "run_id": run_id,
            "video_id": video_id,
            **metrics,
        }
    )

# %%
# Convert results to DataFrame
results_df = pd.DataFrame(results)
frame_results_df = pd.DataFrame(frame_results)

# Save results
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
frame_results_df.to_csv(RESULTS_DIR / "frame_metrics.csv", index=False)

# %%
# Print summary statistics
print("\nSummary Statistics:")
print(results_df.describe())

# No need to group by video, as each row is already per video
# Save video-level metrics (already per video)
results_df.to_csv(RESULTS_DIR / "video_metrics.csv", index=False)
