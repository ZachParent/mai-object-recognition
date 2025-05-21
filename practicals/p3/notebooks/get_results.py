import re

import pandas as pd
from src.config import RESULTS_DIR


def get_metrics_dfs():
    frame_metrics = pd.read_csv(RESULTS_DIR / "frame_metrics.csv")
    video_metrics = pd.read_csv(RESULTS_DIR / "video_metrics.csv")
    return frame_metrics, video_metrics


def get_runs_df(include_run_configs=False):
    run_dirs = sorted(RESULTS_DIR.glob("run_*"))
    runs_df = pd.DataFrame()
    for run_dir in run_dirs:
        match = re.search(r"run_(\d+)", run_dir.name)
        if not match:
            print(
                f"Warning: Could not extract run_id from directory {run_dir.name}. Skipping."
            )
            continue
        run_id = int(match.group(1))
        for set_name in ["train", "val", "test"]:
            metrics_file = run_dir / f"{set_name}.csv"
            try:
                metrics_df = pd.read_csv(metrics_file)
                metrics_df["run_id"] = run_id
                metrics_df["set"] = set_name
                metrics_df["epoch"] = metrics_df.index
                runs_df = pd.concat([runs_df, metrics_df], ignore_index=True)
            except Exception as e:
                print(
                    f"Warning: Could not process file {metrics_file}. Error: {e}. Skipping."
                )
    if include_run_configs:
        run_config_df = pd.read_csv(RESULTS_DIR / "run_configs.csv")
        run_config_df.rename(columns={"id": "run_id"}, inplace=True)
        runs_df = pd.merge(runs_df, run_config_df, on="run_id", how="left")
    return runs_df


if __name__ == "__main__":
    runs_df = get_runs_df()
    print(runs_df)
