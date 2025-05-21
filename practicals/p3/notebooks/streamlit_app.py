import sys
from pathlib import Path

# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from notebooks.get_results import get_metrics_dfs, get_runs_df
from notebooks.quantitative_analysis import (
    plot_combined_metric_distribution,
    plot_correlation_heatmap,
    plot_metric_violin,
)
from src.config import CHECKPOINTS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR
from src.datasets.cloth3d import Cloth3dDataset
from src.inferrer import load_model
from streamlit_image_comparison import image_comparison

# Set page config
st.set_page_config(
    page_title="Depth Estimation Visualization", page_icon="📊", layout="wide"
)

# Title and description
st.title("P3 Depth Estimation")

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_path = str(CHECKPOINTS_DIR / f"run_{0:03d}.pt")
inferrer = load_model(model_path)
raw_dataset = Cloth3dDataset(start_idx=0, enable_normalization=False)
normalized_dataset = Cloth3dDataset(start_idx=0, enable_normalization=True)

# Create tabs for different visualization modes
tab1, tab2, tab3, tab4 = st.tabs(
    ["Single Frame", "GIF Creation", "Quantitative Analysis", "Model Performance"]
)

with tab1:
    st.header("Single Frame Depth Visualization")
    st.markdown(
        """
    Visualize depth estimation results from our trained model.
    Select a video ID and frame ID to see the model's predictions.
    """
    )
    # Load metrics data
    metrics_path = Path(RESULTS_DIR / "frame_metrics.csv")
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

        # Apply filters
        filtered_df = df.copy()
        if video_filter:
            filtered_df = filtered_df[filtered_df["video_id"].isin(video_filter)]

        # Display the table
        event = st.dataframe(
            filtered_df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Add visualization for selected row
        st.subheader("Visualize Selected Frame")
        rows = event.get("selection", {}).get("rows", [])
        selected_row = rows[0] if rows else None

        if selected_row is not None:
            video_id = int(filtered_df.iloc[selected_row]["video_id"])
            frame_id = int(filtered_df.iloc[selected_row]["frame_id"])

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
            fig = inferrer.visualize_prediction(
                video_id=video_id,
                frame_id=frame_id,
                raw_dataset=raw_dataset,
                normalized_dataset=normalized_dataset,
            )

            # Display the plot in Streamlit
            st.pyplot(fig)
            plt.close()

            # Add image comparison widget
            st.subheader("Compare Predicted Depth with Ground Truth")

            # Convert depth maps to RGB for comparison
            def depth_to_rgb(depth_map):
                # Convert to RGB using viridis colormap
                depth_rgb = np.clip(depth_map, 0, 1)
                depth_rgb = plt.get_cmap("viridis")(depth_map)[
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
            )
        else:
            st.warning("Please select a row to visualize.")
    else:
        st.warning(
            "No metrics data found. Please run the quantitative_analysis.py notebook first."
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
    # Load data
    try:
        frame_metrics, video_metrics = get_metrics_dfs()

        if frame_metrics.empty or video_metrics.empty:
            st.warning(
                "Metrics files are empty. Please ensure 'frame_metrics.csv' and 'video_metrics.csv' contain data."
            )
        else:
            # Get metric columns
            metric_columns = [
                col
                for col in frame_metrics.columns
                if col not in ["run_id", "video_id", "frame_id"]
            ]
            if not metric_columns:
                st.warning(
                    "No metric columns found in the data (excluding id columns)."
                )

            st.subheader("Metric Correlation Heatmaps")

            col_heatmap1, col_heatmap2 = st.columns(2)
            with col_heatmap1:
                st.markdown("##### Frame-level Metrics")
                plot_correlation_heatmap(
                    frame_metrics, "Frame-level Metrics Correlation"
                )
                st.pyplot(plt.gcf())
                plt.clf()  # Clear the figure for the next plot

            with col_heatmap2:
                st.markdown("##### Video-level Metrics")
                plot_correlation_heatmap(
                    video_metrics, "Video-level Metrics Correlation"
                )
                st.pyplot(plt.gcf())
                plt.clf()  # Clear the figure

            if metric_columns:
                st.subheader("Metric Distributions per Run")
                for metric in metric_columns:  # Iterate through all metrics
                    st.markdown(
                        f"---"
                    )  # Add a visual separator for each metric's section
                    st.markdown(f"### Distributions for: `{metric}`")

                    # Combined split violin plot
                    plot_combined_metric_distribution(
                        frame_metrics, video_metrics, metric
                    )
                    st.pyplot(plt.gcf())
                    plt.clf()

                    # Individual violin plots side-by-side
                    with st.expander("Individual Violin Plots"):
                        col_dist_violin1, col_dist_violin2 = st.columns(2)
                        with col_dist_violin1:
                            st.markdown(f"##### Frame-level: `{metric}`")
                            plot_metric_violin(
                                frame_metrics,
                                metric,
                                f"Frame-level: {metric} Distribution",  # Simplified title
                            )
                            st.pyplot(plt.gcf())
                            plt.clf()

                        with col_dist_violin2:
                            st.markdown(f"##### Video-level: `{metric}`")
                            plot_metric_violin(
                                video_metrics,
                                metric,
                                f"Video-level: {metric} Distribution",  # Simplified title
                            )
                            st.pyplot(plt.gcf())
                            plt.clf()
            else:
                st.info("No metrics available to display distributions.")

    except FileNotFoundError:
        st.error(
            "Metrics files (frame_metrics.csv or video_metrics.csv) not found in the results directory. "
            "Please run the `get_individual_metrics.py` script first to generate them."
        )
    except Exception as e:
        st.error(f"An error occurred while loading or plotting data: {e}")


def training_results_df(runs_df):
    runs_df["run_id_group"] = runs_df["run_id"].apply(lambda x: x // 3)
    # Group by 'run_id_group' and 'epoch', then aggregate 'mse'
    runs_by_group_df = (
        runs_df.groupby(["run_id_group", "epoch", "set"])["mse"]
        .agg(mse_median="median", mse_min="min", mse_max="max")
        .reset_index()
    )

    set_name = st.selectbox("Set", ["train", "val", "test"], index=2)
    runs_by_group_df = runs_by_group_df[runs_by_group_df["set"] == set_name]
    return runs_by_group_df


def plot_training_curves(runs_df):
    train_and_val_df = runs_df[runs_df["set"].isin(["train", "val"])]
    train_and_val_df["run_id_group"] = train_and_val_df["run_id"].apply(
        lambda x: x // 3
    )
    # Group by 'run_id_group' and 'epoch', then aggregate 'mse'
    train_and_val_by_group_df = (
        train_and_val_df.groupby(["run_id_group", "epoch", "set"])["mse"]
        .agg(mse_median="median", mse_min="min", mse_max="max")
        .reset_index()
    )
    plot = px.line(
        train_and_val_by_group_df,
        x="epoch",
        y="mse_median",
        color="run_id_group",
        facet_col="set",
        error_y_minus="mse_min",
        error_y="mse_max",
    )
    return plot


with tab4:
    st.header("Model Performance")
    runs_df = get_runs_df()
    runs_by_group_df = training_results_df(runs_df)
    st.dataframe(
        runs_by_group_df,
        hide_index=True,
        column_config={
            "mse_median": st.column_config.NumberColumn(format="%.6f"),
            "mse_min": st.column_config.NumberColumn(format="%.6f"),
            "mse_max": st.column_config.NumberColumn(format="%.6f"),
        },
    )
    plot = plot_training_curves(runs_df)
    st.plotly_chart(plot)


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
1. Select a frame from the Single Frame tab
2. View the predicted depth map and the ground truth depth map
3. Compare the prediction and ground truth with the image comparison widget
4. Use the GIF Creation tab to create animations
"""
)
