from src.run_configs import VIT_RUN_SET, RunSet
from src.trainer import run_experiment

group_id = 0
RUN_SET = RunSet(
    title=VIT_RUN_SET.title,
    configs=VIT_RUN_SET.configs[group_id * 3 : (group_id + 1) * 3],
)

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
