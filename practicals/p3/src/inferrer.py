from typing import Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CHECKPOINTS_DIR
from .datasets.cloth3d import Cloth3dDataset
from .models import get_model
from .run_configs import ModelName, RunConfig, UNet2DConfig


class Inferrer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the inferrer with a trained model.

        Args:
            model_path: Path to the saved model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()  # Set model to evaluation mode

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load a trained model from disk.

        Args:
            model_path: Path to the saved model weights

        Returns:
            Loaded model
        """
        # Create a default config for the model
        config = RunConfig(
            id=0,
            name="inference",
            model_name=ModelName.UNET2D,
            learning_rate=0.0001,  # Not used for inference
            unet2d_config=UNet2DConfig(),
        )

        # Initialize model with config
        model = get_model(config)

        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)

        return model

    def infer(
        self,
        video_id: int,
        frame_id: int,
        dataset: Optional[Cloth3dDataset] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make a prediction for a specific video and frame.

        Args:
            video_id: ID of the video
            frame_id: ID of the frame within the video
            dataset: Optional dataset to use for loading the input image

        Returns:
            Tuple of (input_image, predicted_depth, ground_truth_depth)
        """
        if dataset is None:
            # Create a dataset that includes the test set
            dataset = Cloth3dDataset(start_idx=128 + 16, end_idx=None)

        # Calculate the index in the dataset
        idx = video_id * 100 + frame_id  # Assuming 100 frames per video

        # Get input image and ground truth
        input_image, ground_truth = dataset[idx]

        # Add batch dimension and move to device
        input_image = input_image.unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            predicted_depth = self.model(input_image)

        return input_image[0], predicted_depth[0], ground_truth

    def visualize_prediction(
        self,
        video_id: int,
        frame_id: int,
        dataset: Optional[Cloth3dDataset] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize the model's prediction for a specific video and frame.

        Args:
            video_id: ID of the video
            frame_id: ID of the frame within the video
            dataset: Optional dataset to use for loading the input image
            save_path: Optional path to save the visualization
        """
        input_image, predicted_depth, ground_truth = self.infer(
            video_id, frame_id, dataset
        )

        # Move tensors to CPU and convert to numpy
        input_image = input_image.cpu().numpy()
        predicted_depth = predicted_depth.cpu().numpy()
        ground_truth = ground_truth.cpu().numpy()

        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Plot input image
        ax1.imshow(np.transpose(input_image, (1, 2, 0)))
        ax1.set_title("Input Image")
        ax1.axis("off")

        # Plot predicted depth
        im2 = ax2.imshow(predicted_depth[0], cmap="viridis")
        ax2.set_title("Predicted Depth")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2)

        # Plot ground truth depth
        im3 = ax3.imshow(ground_truth[0], cmap="viridis")
        ax3.set_title("Ground Truth Depth")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3)

        plt.suptitle(f"Video {video_id}, Frame {frame_id}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def create_prediction_gif(
        self,
        video_id: int,
        frame_ids: list[int],
        output_path: str,
        fps: int = 10,
        dpi: int = 100,
        dataset: Optional[Cloth3dDataset] = None,
    ) -> None:
        """Create a GIF from a sequence of predictions.

        Args:
            video_id: ID of the video
            frame_ids: List of frame IDs to include in the GIF
            output_path: Path to save the GIF
            fps: Frames per second
            dpi: Dots per inch for the output image
            dataset: Optional dataset to use for loading the input images
        """
        # Create figure and axes
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Initialize empty images
        im1 = ax1.imshow(np.zeros((256, 256, 3)))
        im2 = ax2.imshow(np.zeros((256, 256)), cmap="viridis")
        im3 = ax3.imshow(np.zeros((256, 256)), cmap="viridis")

        # Add colorbars
        plt.colorbar(im2, ax=ax2)
        plt.colorbar(im3, ax=ax3)

        # Set titles and remove axes
        ax1.set_title("Input Image")
        ax2.set_title("Predicted Depth")
        ax3.set_title("Ground Truth Depth")
        for ax in [ax1, ax2, ax3]:
            ax.axis("off")

        def update(frame_id):
            # Get prediction for this frame
            input_image, predicted_depth, ground_truth = self.infer(
                video_id=video_id, frame_id=frame_id, dataset=dataset
            )

            # Move tensors to CPU and convert to numpy
            input_image = input_image.cpu().numpy()
            predicted_depth = predicted_depth.cpu().numpy()
            ground_truth = ground_truth.cpu().numpy()

            # Update images
            im1.set_array(np.transpose(input_image, (1, 2, 0)))
            im2.set_array(predicted_depth[0])
            im3.set_array(ground_truth[0])

            # Update title
            plt.suptitle(f"Video {video_id}, Frame {frame_id}")

            return [im1, im2, im3]

        # Create animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=frame_ids,
            interval=1000 // fps,  # Convert fps to milliseconds
            blit=True,
        )

        # Save animation as GIF
        anim.save(
            output_path,
            writer="pillow",
            fps=fps,
            dpi=dpi,
        )

        plt.close()
        print(f"Saved GIF to {output_path}")


def load_model(model_path: str) -> Inferrer:
    """Helper function to create an Inferrer instance.

    Args:
        model_path: Path to the saved model weights

    Returns:
        Inferrer instance
    """
    return Inferrer(model_path)


if __name__ == "__main__":
    # Example usage
    model_path = str(CHECKPOINTS_DIR / "run_0.pt")
    inferrer = load_model(model_path)

    # Visualize a prediction
    inferrer.visualize_prediction(video_id=0, frame_id=0)
