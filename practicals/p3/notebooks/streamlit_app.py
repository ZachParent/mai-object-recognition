import sys
from pathlib import Path

# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from src.config import CHECKPOINTS_DIR, VISUALIZATIONS_DIR
from src.datasets.cloth3d import Cloth3dDataset
from src.inferrer import load_model

# Set page config
st.set_page_config(
    page_title="Depth Estimation Visualization", page_icon="📊", layout="wide"
)

# Title and description
st.title("Depth Estimation Visualization")
st.markdown(
    """
This app allows you to visualize depth estimation results from our trained model.
Select a video ID and frame ID to see the model's predictions.
"""
)

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_path = str(CHECKPOINTS_DIR / "run_0.pt")
inferrer = load_model(model_path)
dataset = Cloth3dDataset(start_idx=0)

# Main content
st.header("Input Parameters")

# Create tabs for different visualization modes
tab1, tab2 = st.tabs(["Single Frame", "GIF Creation"])

with tab1:
    # Create two columns for input controls
    col1, col2 = st.columns(2)

    with col1:
        video_id = st.number_input(
            "Video ID",
            min_value=0,
            max_value=100,
            value=0,
            help="Select the video ID to visualize",
        )

    with col2:
        frame_id = st.number_input(
            "Frame ID",
            min_value=0,
            max_value=100,
            value=0,
            help="Select the frame ID within the video",
        )

    # Add a button to trigger visualization
    if st.button("Visualize Frame", type="primary"):
        # Create a placeholder for the plot
        plot_placeholder = st.empty()

        # Get the prediction
        input_image, predicted_depth, ground_truth = inferrer.infer(
            video_id=video_id, frame_id=frame_id, dataset=dataset
        )

        # Move tensors to CPU and convert to numpy
        input_image = input_image.cpu().numpy()
        predicted_depth = predicted_depth.cpu().numpy()
        ground_truth = ground_truth.cpu().numpy()

        # Create the visualization
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

        # Display the plot in Streamlit
        plot_placeholder.pyplot(fig)
        plt.close()

with tab2:
    st.header("Create GIF Animation")

    # GIF creation controls
    col1, col2, col3 = st.columns(3)

    with col1:
        gif_video_id = st.number_input(
            "Video ID for GIF",
            min_value=0,
            max_value=100,
            value=0,
            help="Select the video ID for the GIF",
        )

    with col2:
        start_frame = st.number_input(
            "Start Frame",
            min_value=0,
            max_value=100,
            value=0,
            help="Starting frame number",
        )

        end_frame = st.number_input(
            "End Frame",
            min_value=0,
            max_value=100,
            value=100,
            help="Ending frame number",
        )

    with col3:
        frame_step = st.number_input(
            "Frame Step",
            min_value=1,
            max_value=20,
            value=5,
            help="Step between frames (e.g., 5 means every 5th frame)",
        )

        fps = st.number_input(
            "FPS",
            min_value=1,
            max_value=30,
            value=10,
            help="Frames per second in the GIF",
        )

    if st.button("Create GIF", type="primary"):
        # Create frame list
        frame_ids = list(range(start_frame, end_frame + 1, frame_step))

        # Create output path
        output_path = str(VISUALIZATIONS_DIR / f"video_{gif_video_id}_animation.gif")

        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create GIF
        status_text.text("Creating GIF...")
        inferrer.create_prediction_gif(
            video_id=gif_video_id,
            frame_ids=frame_ids,
            output_path=output_path,
            fps=fps,
            dpi=100,
            dataset=dataset,
        )

        # Show success message
        status_text.text("GIF created successfully!")
        st.success(f"GIF saved to: {output_path}")

        # Display the GIF
        st.image(output_path, caption=f"Video {gif_video_id} Animation")

# Add some information about the model
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    """
This app uses a UNet2D model trained on the Cloth3D dataset for depth estimation.
The model takes RGB images as input and predicts depth maps.
"""
)

# Add instructions
st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
1. Select a video ID (0-100)
2. Select a frame ID (0-100)
3. Click 'Visualize' to see the results
4. Use the GIF Creation tab to create animations
"""
)
