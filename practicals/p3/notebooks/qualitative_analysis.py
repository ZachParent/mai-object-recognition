# %%
import sys
from pathlib import Path

# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

from src.config import CHECKPOINTS_DIR, VISUALIZATIONS_DIR
from src.datasets.cloth3d import Cloth3dDataset
from src.inferrer import load_model

# %%
# Load the trained model
model_path = str(CHECKPOINTS_DIR / "run_0.pt")
inferrer = load_model(model_path)

# Create both raw and normalized datasets
raw_dataset = Cloth3dDataset(start_idx=0, enable_normalization=False)
normalized_dataset = Cloth3dDataset(start_idx=0, enable_normalization=True)

VISUALIZATIONS_DIR.mkdir(exist_ok=True, parents=True)

# %%
# Visualize predictions for different videos and frames
video_ids = [1]
frame_ids = [0, 25, 50, 75]

for video_id in video_ids:
    for frame_id in frame_ids:
        print(f"\nVisualizing Video {video_id}, Frame {frame_id}")
        inferrer.visualize_prediction(
            video_id=video_id,
            frame_id=frame_id,
            raw_dataset=raw_dataset,
            normalized_dataset=normalized_dataset,
        )

# %%
# Compare predictions across different frames in the same video
video_id = 1
frame_ids = range(0, 100, 10)  # Every 10th frame

for frame_id in frame_ids:
    print(f"\nVisualizing Video {video_id}, Frame {frame_id}")
    inferrer.visualize_prediction(
        video_id=video_id,
        frame_id=frame_id,
        raw_dataset=raw_dataset,
        normalized_dataset=normalized_dataset,
    )

# %%
# Save visualizations to disk
video_id = 1
frame_ids = [0, 25, 50, 75]

for frame_id in frame_ids:
    save_path = VISUALIZATIONS_DIR / f"video_{video_id}_frame_{frame_id}.png"
    inferrer.visualize_prediction(
        video_id=video_id,
        frame_id=frame_id,
        save_path=str(save_path),
        raw_dataset=raw_dataset,
        normalized_dataset=normalized_dataset,
    )
    print(f"Saved visualization to {save_path}")


# %%
# Interactive visualization
def visualize_frame(video_id: int, frame_id: int):
    """Helper function to visualize a single frame."""
    inferrer.visualize_prediction(
        video_id=video_id,
        frame_id=frame_id,
        raw_dataset=raw_dataset,
        normalized_dataset=normalized_dataset,
    )


# Example usage:
visualize_frame(video_id=1, frame_id=0)

# %%
# Create a GIF from a sequence of frames
video_id = 1
frame_ids = list(range(0, 100, 5))  # Every 5th frame
output_path = str(VISUALIZATIONS_DIR / f"video_{video_id}_animation.gif")

inferrer.create_prediction_gif(
    video_id=video_id,
    frame_ids=frame_ids,
    output_path=output_path,
    fps=10,
    dpi=100,
    raw_dataset=raw_dataset,
    normalized_dataset=normalized_dataset,
)
