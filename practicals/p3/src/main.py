from src.run_configs import PERCEPTUAL_LOSS_RUN_SET
from src.trainer import run_experiment

RUN_SET = PERCEPTUAL_LOSS_RUN_SET

if __name__ == "__main__":
    width = 100
    print(
        f"Running {len(RUN_SET.configs)} experiments from {RUN_SET.title}".center(width)
    )
    print("-" * width)
    for config in RUN_SET.configs:
        print("=" * width)
        print(f"Running {config.name} with id {config.id}".center(width))
        print("-" * width)
        print()
        run_experiment(config)
