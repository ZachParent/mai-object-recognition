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
from src.inferrer import get_run_config, load_model
from src.metrics import get_metric_collection


def get_metrics_for_run(run_id: int):
    """Get frame and video level metrics for a specific run.

    Args:
        run_id: The ID of the run to evaluate

    Returns:
        tuple: (frame_results_df, video_results_df) containing the metrics DataFrames
    """
    # Load the trained model
    model_path = str(CHECKPOINTS_DIR / f"run_{run_id:03d}.pt")
    run_config = get_run_config(run_id)
    inferrer = load_model(model_path, run_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inferrer.model.to(device)

    video_metric_collection = get_metric_collection(run_config)
    frame_metric_collection = get_metric_collection(run_config)

    # Create dataset
    dataset = Cloth3dDataset(start_idx=0, enable_normalization=True)

    # Initialize lists to store results
    results = []
    frame_results = []

    # Process each video
    for video_id in tqdm(dataset.video_ids, desc=f"Processing videos for run {run_id}"):
        # Get all frames for this video
        video_frames = [
            i for i in range(len(dataset)) if dataset.get_video_id(i) == video_id
        ]

        # Reset metrics for each video
        video_metric_collection.reset()

        # Process each frame
        for frame_idx in tqdm(
            video_frames, desc=f"Processing frames for video {video_id}", leave=False
        ):
            # Get data
            image, depth = dataset[frame_idx]

            # Get prediction
            with torch.no_grad():
                pred_depth = inferrer.model(image.unsqueeze(0).to(device))
                pred_depth = pred_depth.squeeze(0).cpu()

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

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    frame_results_df = pd.DataFrame(frame_results)

    return frame_results_df, results_df


def main():

    run_ids = [29, 2, 16]

    # Process each run
    all_frame_results = []
    all_video_results = []

    for run_id in run_ids:
        print(f"\nProcessing run {run_id}")
        frame_results_df, video_results_df = get_metrics_for_run(run_id)
        all_frame_results.append(frame_results_df)
        all_video_results.append(video_results_df)

    # Combine results
    combined_frame_results = pd.concat(all_frame_results, ignore_index=True)
    combined_video_results = pd.concat(all_video_results, ignore_index=True)

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    combined_frame_results.to_csv(RESULTS_DIR / "frame_metrics.csv", index=False)
    combined_video_results.to_csv(RESULTS_DIR / "video_metrics.csv", index=False)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(combined_video_results.describe())


if __name__ == "__main__":
    main()
