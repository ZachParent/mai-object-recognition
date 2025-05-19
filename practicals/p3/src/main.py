from src.run_configs import HYPERPARAM_RUN_SET
from src.trainer import run_experiment

if __name__ == "__main__":
    print(
        f"Running {len(HYPERPARAM_RUN_SET.configs)} experiments from {HYPERPARAM_RUN_SET.title}".center(
            100
        )
    )
    print("-" * 100)
    for config in HYPERPARAM_RUN_SET.configs:
        print("=" * 100)
        print(f"Running {config.name} with id {config.id}".center(100))
        print("-" * 100)
        print()
        run_experiment(config)
