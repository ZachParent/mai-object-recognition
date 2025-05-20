import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# numpy is not strictly needed as torch handles the math

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import RESULTS_DIR
from src.datasets.cloth3d import Cloth3dDataset


def calculate_pixel_distribution():
    """
    Calculates and saves the mean and standard deviation of pixel values
    for each channel in the Cloth3D training dataset.
    Images are loaded as float32 tensors with values scaled to [0, 1].
    """
    print("Initializing Cloth3D dataset for training split (videos 0-128)...")
    # Training split: videos 0 through 128. end_idx is exclusive.
    # enable_normalization=False and enable_augmentation=False ensures that
    # the NON_NORMALIZED_INPUT_TRANSFORM is used from cloth3d.py,
    # which scales images to [0, 1] float32 type.
    train_dataset = Cloth3dDataset(
        start_idx=0,
        end_idx=128,
        enable_normalization=False,
        enable_augmentation=False,
    )

    if not train_dataset:  # Checks if the dataset has any length
        print("Error: The dataset is empty or could not be loaded.")
        print(
            "Please check PREPROCESSED_DATA_DIR in your config and the dataset indices."
        )
        return

    if len(train_dataset) == 0:
        print("Error: The dataset is empty (length is 0).")
        print(
            "Please check PREPROCESSED_DATA_DIR in your config and the dataset indices."
        )
        return

    print(f"Dataset loaded. Number of images: {len(train_dataset)}")

    dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Initialize accumulators. Using float64 for sums to maintain precision.
    # Input images from Cloth3dDataset (with NON_NORMALIZED_INPUT_TRANSFORM) are (C, H, W), type float32, range [0,1].
    channel_sum = torch.zeros(3, dtype=torch.float64)  # For RGB channels
    channel_sum_sq = torch.zeros(3, dtype=torch.float64)
    total_pixels_processed_per_channel = (
        0  # This will be N in the formulas (total count of pixels per channel)
    )

    print("Calculating pixel statistics...")
    num_images_processed = 0
    for i, (images_batch, _) in enumerate(dataloader):
        # images_batch shape: (B, C, H, W), dtype=torch.float32, range [0,1]
        # Convert to float64 for accumulation to avoid precision loss
        images_batch_f64 = images_batch.to(dtype=torch.float64)

        B, C, H, W = images_batch_f64.shape

        if C != 3:
            print(
                f"Warning: Expected 3 channels (RGB), but found {C} channels in batch {i}. Processing first 3 if available."
            )
            if C < 3:
                print(
                    f"Error: Batch {i} has fewer than 3 channels ({C}). Cannot proceed."
                )
                return
            images_batch_f64 = images_batch_f64[:, :3, :, :]  # Take first 3 channels
            # C is now guaranteed to be 3 for the below logic if C was >3, or error returned if C < 3.

        # Identify black pixels (R=0, G=0, B=0)
        # images_batch_f64 shape is (B, 3, H, W)
        is_r_zero = images_batch_f64[:, 0, :, :] == 0.0
        is_g_zero = images_batch_f64[:, 1, :, :] == 0.0
        is_b_zero = images_batch_f64[:, 2, :, :] == 0.0

        # Mask for pixels that are entirely black across all three channels
        is_black_pixel_bhw = is_r_zero & is_g_zero & is_b_zero  # Shape (B, H, W)

        # Mask for non-black pixels (pixels where at least one channel is non-zero)
        non_black_mask_bhw = ~is_black_pixel_bhw  # Shape (B, H, W), dtype=torch.bool

        # Count number of non-black pixels in this batch (scalar)
        num_non_black_pixels_this_batch = (
            non_black_mask_bhw.sum().item()
        )  # sum() on bool tensor gives int tensor

        if num_non_black_pixels_this_batch > 0:
            # Expand non_black_mask_bhw to apply to each channel for element-wise multiplication
            # (B, H, W) -> (B, 1, H, W) -> (B, 3, H, W)
            non_black_mask_bchw = non_black_mask_bhw.unsqueeze(1).expand_as(
                images_batch_f64
            )

            # Zero out the black pixels by multiplying with the mask (True=1, False=0).
            # This ensures that black pixels contribute 0 to the sums.
            # Values of non-black pixels are preserved.
            # Convert boolean mask to the dtype of images_batch_f64 for multiplication.
            masked_images = images_batch_f64 * non_black_mask_bchw.to(
                images_batch_f64.dtype
            )

            # Sum pixel values from non-black pixels over Batch, Height, and Width dimensions for each channel
            channel_sum += torch.sum(masked_images, dim=(0, 2, 3))  # Result shape: (3,)
            channel_sum_sq += torch.sum(
                masked_images.pow(2), dim=(0, 2, 3)
            )  # Result shape: (3,)

            # Accumulate the count of non-black pixels processed.
            # This count is used as N in mean (Sum(X)/N) and std dev calculations.
            total_pixels_processed_per_channel += num_non_black_pixels_this_batch
        # If num_non_black_pixels_this_batch is 0 for this batch (e.g. all pixels in batch are black),
        # then no values are added to sums, and total_pixels_processed_per_channel is not increased for this batch.

        num_images_processed += B

        if (i + 1) % 50 == 0 or (i + 1) == len(
            dataloader
        ):  # Print progress every 50 batches and for the last batch
            print(
                f"Processed {num_images_processed}/{len(train_dataset)} images ({i+1}/{len(dataloader)} batches)..."
            )

    if total_pixels_processed_per_channel == 0:
        print(
            "Error: No pixels were processed. This might indicate an empty dataset or images of zero size."
        )
        return

    print("Calculation complete.")
    print(f"Total pixels processed (per channel): {total_pixels_processed_per_channel}")

    # Calculate mean and standard deviation
    # Mean = Sum(X_i) / N
    mean_per_channel = channel_sum / total_pixels_processed_per_channel
    # StdDev = sqrt( Sum(X_i^2)/N - Mean^2 )
    std_per_channel = (
        channel_sum_sq / total_pixels_processed_per_channel - mean_per_channel.pow(2)
    ).sqrt()

    print("\n--- Pixel Statistics (Range [0, 1]) ---")
    channel_names = ["Red", "Green", "Blue"]
    for i in range(3):
        print(f"Channel {channel_names[i]}:")
        print(f"  Mean:    {mean_per_channel[i].item():.6f}")
        print(f"  Std Dev: {std_per_channel[i].item():.6f}")

    results_dir = RESULTS_DIR
    results_dir.mkdir(
        parents=True, exist_ok=True
    )  # Create 'results' directory if it doesn't exist
    csv_file_path = results_dir / "pixel_distribution.csv"

    print(f"\nSaving statistics to: {csv_file_path}")
    header = ["channel_name", "mean", "std_dev"]
    data_rows = [
        [
            channel_names[0],
            f"{mean_per_channel[0].item():.6f}",
            f"{std_per_channel[0].item():.6f}",
        ],
        [
            channel_names[1],
            f"{mean_per_channel[1].item():.6f}",
            f"{std_per_channel[1].item():.6f}",
        ],
        [
            channel_names[2],
            f"{mean_per_channel[2].item():.6f}",
            f"{std_per_channel[2].item():.6f}",
        ],
    ]

    try:
        with open(csv_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data_rows)
        print("Statistics saved successfully.")
    except IOError as e:
        print(f"Error: Could not write to CSV file {csv_file_path}.")
        print(f"IOError details: {e}")


if __name__ == "__main__":
    calculate_pixel_distribution()
