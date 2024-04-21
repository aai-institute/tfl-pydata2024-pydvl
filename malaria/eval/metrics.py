from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TypeVar, Generic, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from malaria.eval.evaluator import InfluenceEvaluationInput

T = TypeVar("T")


class EvalMetric(Generic[T], ABC):
    """
    Abstract base class for evaluation metrics.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        """
       Should return the name of the metric.
       """

    @abstractmethod
    def compute(self, eval_input: "InfluenceEvaluationInput") -> List[T]:
        """
        Computes the metric based on the provided InfluenceEvaluationInput.

        Args:
            eval_input: The evaluation input data.

        Returns:
            A list containing the computed metric results.
        """


class MeanInfluence(EvalMetric[float]):
    """
    Metric that calculates the mean influence value across all test data points.
    """

    MEAN_INFLUENCE_METRIC_NAME = "mean_influence"

    @property
    def name(self) -> str:
        return self.MEAN_INFLUENCE_METRIC_NAME

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        return eval_input.influence_value_array.mean(axis=0)


class MeanInfluenceUnInfected(EvalMetric[float]):
    """
    Metric that calculates the mean influence value across all uninfected test
    data points.
    """

    MEAN_INFLUENCE_UNINFECTED_METRIC_NAME = "mean_influence_uninfected"

    @property
    def name(self) -> str:
        return self.MEAN_INFLUENCE_UNINFECTED_METRIC_NAME

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        uninfected_test_indices = eval_input.uninfected_test_indices
        return eval_input.influence_value_array[uninfected_test_indices, :].mean(axis=0)


class MeanInfluenceParasitized(EvalMetric[float]):
    """
    Metric that calculates the mean influence value across all parasitized test
    data points.
    """

    MEAN_INFLUENCE_UNINFECTED_METRIC_NAME = "mean_influence_parasitized"

    @property
    def name(self) -> str:
        return self.MEAN_INFLUENCE_UNINFECTED_METRIC_NAME

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        parasitized_test_indices = eval_input.parasitized_test_indices
        return eval_input.influence_value_array[parasitized_test_indices, :].mean(
            axis=0
        )


class QuantileInfluence(EvalMetric):
    """
    Metric that calculates a specific quantile of the influence values across all test
    data points.
    """

    QUANTILE_INFLUENCE_SUFFIX = "_quantile_influence"

    def __init__(self, quantile: float):
        self.quantile = quantile

    @property
    def name(self) -> str:
        return f"{self.quantile}{self.QUANTILE_INFLUENCE_SUFFIX}"

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        return np.quantile(eval_input.influence_value_array, self.quantile, axis=0)


class QuantileInfluenceUninfected(EvalMetric):
    """
    Metric that calculates a specific quantile of the influence values across all
    uninfected test data points.
    """

    QUANTILE_INFLUENCE_SUFFIX = "_quantile_influence_uninfected"

    def __init__(self, quantile: float):
        self.quantile = quantile

    @property
    def name(self) -> str:
        return f"{self.quantile}{self.QUANTILE_INFLUENCE_SUFFIX}"

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        uninfected_test_indices = eval_input.uninfected_test_indices
        uninfected_array = eval_input.influence_value_array[uninfected_test_indices, :]
        return np.quantile(uninfected_array, self.quantile, axis=0)


class QuantileInfluenceParasitized(EvalMetric):
    """
    Metric that calculates a specific quantile of the influence values across all
    parasitized test data points.
    """

    QUANTILE_INFLUENCE_SUFFIX = "_quantile_influence_parasitized"

    def __init__(self, quantile: float):
        self.quantile = quantile

    @property
    def name(self) -> str:
        return f"{self.quantile}{self.QUANTILE_INFLUENCE_SUFFIX}"

    def compute(self, eval_input: InfluenceEvaluationInput) -> List[float]:
        parasitized_test_indices = eval_input.parasitized_test_indices
        parasitized_array = eval_input.influence_value_array[
            parasitized_test_indices, :
        ]
        return np.quantile(parasitized_array, self.quantile, axis=0)


class DefaultMetrics(Enum):
    MeanInfluence = MeanInfluence()
    MeanInfluenceUnInfected = MeanInfluenceUnInfected()
    MeanInfluenceParasitized = MeanInfluenceParasitized()
    QuantileInfluence0_25 = QuantileInfluence(0.25)
    QuantileInfluenceUninfected0_25 = QuantileInfluenceUninfected(0.25)
    QuantileInfluenceParasitized0_25 = QuantileInfluenceParasitized(0.25)
