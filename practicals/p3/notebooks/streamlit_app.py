import sys
from pathlib import Path
from typing import List, Literal

# Add src to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "src"))

import io

import pandas as pd
import plotly.express as px
import streamlit as st
from notebooks.get_results import get_metrics_dfs, get_runs_df
from PIL import Image
from src.config import CHECKPOINTS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR
from src.datasets.cloth3d import Cloth3dDataset
from src.inferrer import get_run_config, load_model
from streamlit_image_comparison import image_comparison

# Set page config
st.set_page_config(
    page_title="Depth Estimation Visualization", page_icon="📊", layout="wide"
)

# Title and description
st.title("P3 Depth Estimation")


@st.cache_data
def load_model_cached(model_id: int):
    config = get_run_config(model_id)
    model_path = str(CHECKPOINTS_DIR / f"run_{model_id:03d}.pt")
    return load_model(model_path, config)


raw_dataset = Cloth3dDataset(start_idx=0, enable_normalization=False)
normalized_dataset = Cloth3dDataset(start_idx=0, enable_normalization=True)


@st.cache_data
def get_metrics_df_cached():
    if not Path(RESULTS_DIR / "frame_metrics.csv").exists():
        st.error(
            "Frame metrics file not found. Please run the `get_individual_metrics.py` script first to generate it."
        )
    return pd.read_csv(Path(RESULTS_DIR / "frame_metrics.csv"))


@st.cache_data
def get_metrics_dfs_cached():
    return get_metrics_dfs()


def save_image(fig, path):
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    img.save(path)


def save_images(figs, run_id, video_id, frame_id):
    fig_labels = ["raw_input", "normalized_input", "predicted_depth", "ground_truth"]
    for i, fig in enumerate(figs):
        save_image(
            fig,
            Path(VISUALIZATIONS_DIR)
            / f"run_{run_id:03d}_video_{video_id}_frame_{frame_id}_{fig_labels[i]}.png",
        )
    st.toast("Images saved")


@st.fragment
def display_single_frame_depth_visualization():
    st.header("Single Frame Depth Visualization")
    st.markdown(
        """
    Visualize depth estimation results from our trained model.
    Select a video ID and frame ID to see the model's predictions.
    """
    )
    model_ids = sorted(
        int(path.stem.split("_")[1]) for path in CHECKPOINTS_DIR.glob("run_*.pt")
    )

    # Load metrics data
    df = get_metrics_df_cached()

    # Add filters
    col1, col2 = st.columns(2)

    with col1:
        video_filter = st.multiselect(
            "Filter by Video ID",
            options=sorted(df["video_id"].unique()),
            default=[],
        )
    with col2:
        model_id = st.selectbox("Model", sorted(df["run_id"].unique()), index=0)

    try:
        inferrer = load_model_cached(model_id)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Apply filters
    filtered_df = df.copy()
    filtered_df = filtered_df[filtered_df["run_id"] == model_id]
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
        raw_input_image = raw_input_image.cpu().numpy().transpose(1, 2, 0)
        normalized_input_image = normalized_input_image.cpu().numpy().transpose(1, 2, 0)
        predicted_depth = predicted_depth.cpu()[0].numpy()
        ground_truth = ground_truth.cpu()[0].numpy()

        figs = [
            px.imshow(raw_input_image),
            px.imshow(normalized_input_image, zmin=0, zmax=1),
            px.imshow(predicted_depth, color_continuous_scale="tempo", zmin=0, zmax=1),
            px.imshow(ground_truth, color_continuous_scale="tempo", zmin=0, zmax=1),
        ]
        cols = st.columns(len(figs))
        for i, fig in enumerate(figs):
            with cols[i]:
                fig.update_layout(
                    coloraxis_showscale=False,
                    xaxis_visible=False,
                    yaxis_visible=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    width=700,
                    height=700,
                    autosize=False,
                )
                st.plotly_chart(fig)
                st.markdown(
                    f"**{['Raw Input Image', 'Normalized Input Image', 'Predicted Depth', 'Ground Truth'][i]}**"
                )
        if st.button("Save Images", type="primary"):
            save_images(figs, model_id, video_id, frame_id)

        # Add image comparison widget
        st.subheader("Compare Predicted Depth with Ground Truth")

        # Update layout to remove axes and colorbar
        imgs = []
        for fig in [figs[2], figs[3]]:
            fig_bytes = fig.to_image(format="png")
            buf = io.BytesIO(fig_bytes)
            img = Image.open(buf)
            imgs.append(img)
        # Create image comparison using Plotly figures
        image_comparison(
            img1=imgs[0],
            img2=imgs[1],
            label1=f"Predicted Depth (model {model_id})",
            label2="Ground Truth",
            in_memory=True,
        )
    else:
        st.warning("Please select a row to visualize.")


def display_training_results_df(runs_df):
    runs_by_group_df = (
        runs_df.groupby(["name", "set", "epoch"])["mse"]
        .agg(mse_median="median", mse_min="min", mse_max="max")
        .reset_index()
    )

    set_name = st.selectbox("Set", ["train", "val", "test"], index=2)
    runs_by_group_df = runs_by_group_df[runs_by_group_df["set"] == set_name]
    st.dataframe(
        runs_by_group_df,
        hide_index=True,
        column_config={
            "mse_median": st.column_config.NumberColumn(format="%.6f"),
            "mse_min": st.column_config.NumberColumn(format="%.6f"),
            "mse_max": st.column_config.NumberColumn(format="%.6f"),
        },
    )


def plot_training_curves(
    runs_df, metric_name: str, sets: List[Literal["train", "val"]], run_sets: List[str]
):
    color_discrete_map = {
        name: px.colors.qualitative.Dark24[i % len(px.colors.qualitative.Dark24)]
        for i, name in enumerate(runs_df["name"].unique())
    }
    # st.write(color_discrete_map)
    train_and_val_df = runs_df[
        runs_df["set"].isin(sets) & runs_df["run_set"].isin(run_sets)
    ]
    train_and_val_by_group_df = (
        train_and_val_df.groupby(["run_set", "name", "set", "epoch"])[metric_name]
        .agg(median="median", min="min", max="max")
        .reset_index()
    )
    train_and_val_by_group_df["error_y"] = (
        train_and_val_by_group_df["max"] - train_and_val_by_group_df["median"]
    )
    train_and_val_by_group_df["error_y_minus"] = (
        train_and_val_by_group_df["max"] - train_and_val_by_group_df["median"]
    )
    train_and_val_by_group_df.rename(columns={"median": metric_name}, inplace=True)
    plot = px.line(
        train_and_val_by_group_df,
        x="epoch",
        y=metric_name,
        log_y=True,
        color="name",
        facet_col="set",
        error_y_minus="error_y_minus",
        error_y="error_y",
        color_discrete_map=color_discrete_map,
        height=600,
    )
    return plot


@st.cache_data
def get_runs_df_cached():
    return get_runs_df(include_run_configs=True)


@st.fragment
def display_model_performance():
    st.header("Model Performance")
    runs_df = get_runs_df_cached()

    run_sets = runs_df["run_set"].unique()

    col1, col2 = st.columns(2)
    with col1:
        metric_name = st.selectbox(
            "Metric", ["mse", "mae", "perceptual_l1", "perceptual_l2"], index=0
        )
    with col2:
        enabled_sets: List[Literal["train", "val"]] = st.multiselect(
            "Sets", ["train", "val"], default=["val"]
        )
    enabled_run_sets = st.pills(
        "Run Sets",
        run_sets,
        selection_mode="multi",
        default=run_sets,
    )
    plot = plot_training_curves(runs_df, metric_name, enabled_sets, enabled_run_sets)
    st.plotly_chart(plot)

    display_training_results_df(runs_df)


# Create tabs for different visualization modes
tab1, tab2 = st.tabs(["Model Performance", "Single Frame"])


with tab1:
    display_model_performance()

with tab2:
    display_single_frame_depth_visualization()
