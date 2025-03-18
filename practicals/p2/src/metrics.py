from typing import Callable, Literal
import torch


class Metric(Callable):
    def __init__(self):
        self.name: str | None = None
        self.display_name: str | None = None

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        pass


class MDice(Metric):
    def __init__(self):
        super().__init__()
        self.name: Literal["m_dice"] = "m_dice"
        self.display_name: Literal["mDice"] = "mDice"

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return (outputs.argmax(dim=1) == targets).float().mean().item()


class F1Score(Metric):
    def __init__(self):
        super().__init__()
        self.name = "f1_score"
        self.display_name = "F1 Score"

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        return (outputs.argmax(dim=1) == targets).float().mean().item()


ALL_METRICS = [MDice(), F1Score()]
