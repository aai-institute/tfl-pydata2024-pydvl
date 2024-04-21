from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from malaria.eval.evaluator import InfluenceEvaluationInput

T = TypeVar("T")

UNINFECTED_VAL = 1
PARASITIZED_VAL = 0


class EvalMetric(Generic[T], ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, eval_input: "InfluenceEvaluationInput") -> List[T]:
        pass


class MeanInfluence(EvalMetric[float]):
    MEAN_INFLUENCE_METRIC_NAME = "mean_influence"

    @property
    def name(self) -> str:
        return self.MEAN_INFLUENCE_METRIC_NAME

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        return eval_input.influence_value_array.mean(axis=0)


class MeanInfluenceUnInfected(EvalMetric[float]):
    MEAN_INFLUENCE_UNINFECTED_METRIC_NAME = "mean_influence_uninfected"

    @property
    def name(self) -> str:
        return self.MEAN_INFLUENCE_UNINFECTED_METRIC_NAME

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        uninfected_test_indices = [
            idx
            for idx, value in enumerate(eval_input.test_labels)
            if value == UNINFECTED_VAL
        ]
        return eval_input.influence_value_array[uninfected_test_indices, :].mean(axis=0)


class MeanInfluenceParasitized(EvalMetric[float]):
    MEAN_INFLUENCE_UNINFECTED_METRIC_NAME = "mean_influence_parasitized"

    @property
    def name(self) -> str:
        return self.MEAN_INFLUENCE_UNINFECTED_METRIC_NAME

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        parasitized_test_indices = [
            idx
            for idx, value in enumerate(eval_input.test_labels)
            if value == PARASITIZED_VAL
        ]
        return eval_input.influence_value_array[parasitized_test_indices, :].mean(
            axis=0
        )


class QuantileInfluence(EvalMetric):
    QUANTILE_INFLUENCE_SUFFIX = "_quantile_influence"

    def __init__(self, quantile: float):
        self.quantile = quantile

    @property
    def name(self) -> str:
        return f"{self.quantile}{self.QUANTILE_INFLUENCE_SUFFIX}"

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        return np.quantile(eval_input.influence_value_array, self.quantile, axis=0)


class QuantileInfluenceUninfected(EvalMetric):
    QUANTILE_INFLUENCE_SUFFIX = "_quantile_influence_uninfected"

    def __init__(self, quantile: float):
        self.quantile = quantile

    @property
    def name(self) -> str:
        return f"{self.quantile}{self.QUANTILE_INFLUENCE_SUFFIX}"

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        uninfected_test_indices = [
            idx
            for idx, value in enumerate(eval_input.test_labels)
            if value == UNINFECTED_VAL
        ]
        uninfected_array = eval_input.influence_value_array[uninfected_test_indices, :]
        return np.quantile(uninfected_array, self.quantile, axis=0)


class QuantileInfluenceParasitized(EvalMetric):
    QUANTILE_INFLUENCE_SUFFIX = "_quantile_influence_parasitized"

    def __init__(self, quantile: float):
        self.quantile = quantile

    @property
    def name(self) -> str:
        return f"{self.quantile}{self.QUANTILE_INFLUENCE_SUFFIX}"

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        parasitized_test_indices = [
            idx
            for idx, value in enumerate(eval_input.test_labels)
            if value == UNINFECTED_VAL
        ]
        parasitized_array = eval_input.influence_value_array[
            parasitized_test_indices, :
        ]
        return np.quantile(parasitized_array, self.quantile, axis=0)


DEFAULT_METRICS = [
    MeanInfluence(),
    MeanInfluenceUnInfected(),
    MeanInfluenceParasitized(),
    QuantileInfluence(0.05),
    QuantileInfluenceUninfected(0.05),
    QuantileInfluenceParasitized(0.05),
    QuantileInfluence(0.10),
    QuantileInfluenceUninfected(0.10),
    QuantileInfluenceParasitized(0.10),
    QuantileInfluence(0.25),
    QuantileInfluenceUninfected(0.25),
    QuantileInfluenceParasitized(0.25),
]


class DefaultMetrics(Enum):
    MeanInfluence = MeanInfluence()
    MeanInfluenceUnInfected = MeanInfluenceUnInfected()
    MeanInfluenceParasitized = MeanInfluenceParasitized()
    QuantileInfluence0_25 = QuantileInfluence(0.25)
    QuantileInfluenceUninfected0_25 = QuantileInfluenceUninfected(0.25)
    QuantileInfluenceParasitized0_25 = QuantileInfluenceParasitized(0.25)
