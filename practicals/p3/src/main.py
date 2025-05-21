from src.run_configs import HYPERPARAM_RUN_SET
from src.trainer import run_experiment

if __name__ == "__main__":
    width = 100
    print(
        f"Running {len(HYPERPARAM_RUN_SET.configs)} experiments from {HYPERPARAM_RUN_SET.title}".center(
            width
        )
    )
    print("-" * width)
    for config in HYPERPARAM_RUN_SET.configs:
        print("=" * width)
        print(f"Running {config.name} with id {config.id}".center(width))
        print("-" * width)
        print()
        run_experiment(config)
