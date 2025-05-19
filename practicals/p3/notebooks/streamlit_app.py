import sys
from pathlib import Path

# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from src.config import CHECKPOINTS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR
from src.datasets.cloth3d import Cloth3dDataset
from src.inferrer import load_model
from streamlit_image_comparison import image_comparison

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
raw_dataset = Cloth3dDataset(start_idx=0, enable_normalization=False)
normalized_dataset = Cloth3dDataset(start_idx=0, enable_normalization=True)

# Main content
st.header("Input Parameters")

# Create tabs for different visualization modes
tab1, tab2, tab3 = st.tabs(["Single Frame", "GIF Creation", "Quantitative Analysis"])

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
        raw_input_image, normalized_input_image, predicted_depth, ground_truth = (
            inferrer.infer(
                video_id=video_id,
                frame_id=frame_id,
                raw_dataset=raw_dataset,
                normalized_dataset=normalized_dataset,
            )
        )

        # Move tensors to CPU and convert to numpy
        raw_input_image = raw_input_image.cpu().numpy()
        normalized_input_image = normalized_input_image.cpu().numpy()
        predicted_depth = predicted_depth.cpu().numpy()
        ground_truth = ground_truth.cpu().numpy()

        # Create the visualization
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

        # Plot input image
        ax1.imshow(np.transpose(raw_input_image, (1, 2, 0)))
        ax1.set_title("Raw Input Image")
        ax1.axis("off")

        # Plot normalized input image
        ax2.imshow(np.transpose(normalized_input_image, (1, 2, 0)))
        ax2.set_title("Normalized Input Image")
        ax2.axis("off")

        # Plot predicted depth
        im3 = ax3.imshow(predicted_depth[0], cmap="viridis")
        ax3.set_title("Predicted Depth")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3)

        # Plot ground truth depth
        im4 = ax4.imshow(ground_truth[0], cmap="viridis")
        ax4.set_title("Ground Truth Depth")
        ax4.axis("off")
        plt.colorbar(im4, ax=ax4)

        plt.suptitle(f"Video {video_id}, Frame {frame_id}")
        plt.tight_layout()

        # Display the plot in Streamlit
        plot_placeholder.pyplot(fig)
        plt.close()

        # Add image comparison widget
        st.subheader("Compare Predicted Depth with Ground Truth")

        # Convert depth maps to RGB for comparison
        def depth_to_rgb(depth_map):
            # Normalize to [0, 1]
            depth_norm = (depth_map - depth_map.min()) / (
                depth_map.max() - depth_map.min() + 1e-8
            )
            # Convert to RGB using viridis colormap
            depth_rgb = plt.get_cmap("viridis")(depth_norm)[
                ..., :3
            ]  # Remove alpha channel
            # Convert to uint8 (0-255)
            depth_rgb = (depth_rgb * 255).astype(np.uint8)
            return depth_rgb

        # Convert depth maps to RGB
        predicted_rgb = depth_to_rgb(predicted_depth[0])
        ground_truth_rgb = depth_to_rgb(ground_truth[0])

        # Create image comparison
        image_comparison(
            img1=predicted_rgb,
            img2=ground_truth_rgb,
            label1="Predicted Depth",
            label2="Ground Truth",
            width=700,
        )

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
            raw_dataset=raw_dataset,
            normalized_dataset=normalized_dataset,
        )

        # Show success message
        status_text.text("GIF created successfully!")
        st.success(f"GIF saved to: {output_path}")
        st.warning("GIFS cannot be played in the app. Please view it in the repo.")

        # Display the GIF
        st.image(output_path, caption=f"Video {gif_video_id} Animation")

with tab3:
    st.header("Quantitative Analysis")

    # Load metrics data
    metrics_path = Path(RESULTS_DIR / "quantitative_analysis.csv")
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)

        # Add filters
        st.subheader("Filters")
        col1, col2 = st.columns(2)

        with col1:
            video_filter = st.multiselect(
                "Filter by Video ID",
                options=sorted(df["video_id"].unique()),
                default=[],
            )

        with col2:
            metric_filter = st.multiselect(
                "Filter by Metric",
                options=[
                    col for col in df.columns if col not in ["video_id", "frame_id"]
                ],
                default=[],
            )

        # Apply filters
        filtered_df = df.copy()
        if video_filter:
            filtered_df = filtered_df[filtered_df["video_id"].isin(video_filter)]
        if metric_filter:
            filtered_df = filtered_df[["video_id", "frame_id"] + metric_filter]

        # Add sorting
        st.subheader("Sorting")
        sort_col = st.selectbox("Sort by", options=filtered_df.columns, index=0)
        sort_asc = st.checkbox("Ascending", value=True)
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

        # Display the table
        st.dataframe(filtered_df, use_container_width=True)

        # Add visualization for selected row
        st.subheader("Visualize Selected Frame")
        selected_row = st.selectbox(
            "Select a row to visualize",
            options=range(len(filtered_df)),
            format_func=lambda x: f"Video {filtered_df.iloc[x]['video_id']}, Frame {filtered_df.iloc[x]['frame_id']}",
        )

        if selected_row is not None:
            row = filtered_df.iloc[selected_row]
            video_id = int(row["video_id"])
            frame_id = int(row["frame_id"])

            # Get the prediction
            raw_input_image, normalized_input_image, predicted_depth, ground_truth = (
                inferrer.infer(
                    video_id=video_id,
                    frame_id=frame_id,
                    raw_dataset=raw_dataset,
                    normalized_dataset=normalized_dataset,
                )
            )

            # Move tensors to CPU and convert to numpy
            raw_input_image = raw_input_image.cpu().numpy()
            normalized_input_image = normalized_input_image.cpu().numpy()
            predicted_depth = predicted_depth.cpu().numpy()
            ground_truth = ground_truth.cpu().numpy()

            # Create the visualization
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

            # Plot input image
            ax1.imshow(np.transpose(raw_input_image, (1, 2, 0)))
            ax1.set_title("Raw Input Image")
            ax1.axis("off")

            # Plot normalized input image
            ax2.imshow(np.transpose(normalized_input_image, (1, 2, 0)))
            ax2.set_title("Normalized Input Image")
            ax2.axis("off")

            # Plot predicted depth
            im3 = ax3.imshow(predicted_depth[0], cmap="viridis")
            ax3.set_title("Predicted Depth")
            ax3.axis("off")
            plt.colorbar(im3, ax=ax3)

            # Plot ground truth depth
            im4 = ax4.imshow(ground_truth[0], cmap="viridis")
            ax4.set_title("Ground Truth Depth")
            ax4.axis("off")
            plt.colorbar(im4, ax=ax4)

            plt.suptitle(f"Video {video_id}, Frame {frame_id}")
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig)
            plt.close()

            # Add image comparison widget
            st.subheader("Compare Predicted Depth with Ground Truth")

            # Convert depth maps to RGB for comparison
            def depth_to_rgb(depth_map):
                # Normalize to [0, 1]
                depth_norm = (depth_map - depth_map.min()) / (
                    depth_map.max() - depth_map.min() + 1e-8
                )
                # Convert to RGB using viridis colormap
                depth_rgb = plt.get_cmap("viridis")(depth_norm)[
                    ..., :3
                ]  # Remove alpha channel
                # Convert to uint8 (0-255)
                depth_rgb = (depth_rgb * 255).astype(np.uint8)
                return depth_rgb

            # Convert depth maps to RGB
            predicted_rgb = depth_to_rgb(predicted_depth[0])
            ground_truth_rgb = depth_to_rgb(ground_truth[0])

            # Create image comparison
            image_comparison(
                img1=predicted_rgb,
                img2=ground_truth_rgb,
                label1="Predicted Depth",
                label2="Ground Truth",
                width=700,
            )
    else:
        st.warning(
            "No metrics data found. Please run the quantitative analysis notebook first."
        )

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
5. Use the Quantitative Analysis tab to explore metrics
"""
)
