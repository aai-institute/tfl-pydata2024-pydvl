from dataclasses import dataclass
import os
import logging
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numpy.typing import NDArray
from torchvision.datasets import ImageFolder
import pandas as pd

from .metrics import EvalMetric, DefaultMetrics
from ..dataset import Label

COL_LABEL = "label"
COL_LABEL_NAME = "label_name"
COL_FILE_PATH = "relative_file_path"


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InfluenceEvaluationInput:
    influence_value_array: NDArray
    dataset: ImageFolder
    test_indices: List[int]
    train_indices: List[int]

    @cached_property
    def train_labels(self) -> List[int]:
        return [self.dataset.targets[idx] for idx in self.train_indices]

    @cached_property
    def uninfected_test_indices(self) -> List[int]:
        uninfected_test_indices = [
            idx
            for idx, value in enumerate(self.test_labels)
            if value == Label.UNINFECTED.value
        ]

        return uninfected_test_indices

    @cached_property
    def parasitized_test_indices(self) -> List[int]:
        parasitized_test_indices = [
            idx
            for idx, value in enumerate(self.test_labels)
            if value == Label.PARASITIZED.value
        ]

        return parasitized_test_indices

    @cached_property
    def test_labels(self) -> List[int]:
        return [self.dataset.targets[idx] for idx in self.test_indices]

    @cached_property
    def train_relative_file_path(self) -> List[Path]:
        root_path = Path(self.dataset.root)
        relative_file_paths = []

        for idx in self.train_indices:
            full_path = Path(self.dataset.imgs[idx][0])
            relative_path = full_path.relative_to(root_path)
            relative_file_paths.append(relative_path)

        return relative_file_paths

    def to_pickle(self, pickle_path: str):
        with open(pickle_path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def from_pickle(pickle_path: str) -> "InfluenceEvaluationInput":
        with open(pickle_path, "rb") as file:
            return pickle.load(file)


@dataclass(frozen=True)
class InfluenceEvaluationResult:
    eval_df: pd.DataFrame
    metric_columns: List[str]

    def generate_hist_plot(
        self,
        metric: Union[str, EvalMetric],
        filter_label: Optional[Label] = None,
        plot_base_path: Optional[str] = None,
        **hist_kwargs,
    ):
        if isinstance(metric, EvalMetric):
            metric_name = metric.name
        else:
            metric_name = metric

        if metric_name not in self.metric_columns:
            raise ValueError(
                f"{metric_name=} not contained in result. "
                f"Available metrics: {self.metric_columns}"
            )

        plot_title = f"Histogram of {metric_name}"
        y_label = "density" if "stat" not in hist_kwargs else hist_kwargs["stat"]

        if filter_label is not None:
            filtered_df = self.eval_df[self.eval_df[COL_LABEL] == filter_label.value]
            plot_title += f": {filter_label.name.lower()} only"
        else:
            filtered_df = self.eval_df

        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_df, x=metric_name, hue=COL_LABEL_NAME, **hist_kwargs)
        plt.title(plot_title)
        plt.xlabel(metric_name)
        plt.ylabel(y_label)
        plt.grid(True)

        if plot_base_path is not None:
            os.makedirs(plot_base_path, exist_ok=True)
            file_path = os.path.join(plot_base_path, f"{metric_name}_{y_label}")

            if filter_label is not None:
                file_path += f"_{filter_label.name.lower()}_only"

            file_path += ".png"

            plt.savefig(file_path, format="png", dpi=300)
            logger.info(f"Plot saved as '{file_path}'")

        plt.show()

    def generate_smallest_k_by_metric(
        self,
        metric: Union[str, EvalMetric],
        smallest_k: int,
        data_cell_base_path: str,
        filter_label: Optional[Label] = None,
        plot_base_path: Optional[str] = None,
    ):
        metric_name = metric.name if isinstance(metric, EvalMetric) else metric

        if metric_name not in self.metric_columns:
            raise ValueError(
                f"{metric_name=} not contained in result. "
                f"Available metrics: {self.metric_columns}"
            )

        if filter_label is not None:
            condition_df = self.eval_df[self.eval_df[COL_LABEL] == filter_label.value]
        else:
            condition_df = self.eval_df

        smallest_k_df = condition_df.nsmallest(smallest_k, metric_name, keep="all")

        # Determine the number of rows and columns for the subplot grid
        n_cols = 3 if smallest_k > 1 else 1
        n_rows = (smallest_k + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 5 * n_rows)
        )
        fig.tight_layout(pad=4.0)

        axes = axes.flatten() if smallest_k > 1 else [axes]

        for ax, (_, row) in zip(axes, smallest_k_df.iterrows()):
            image_path = row[COL_FILE_PATH]
            full_image_path = os.path.join(data_cell_base_path, image_path)
            image = mpimg.imread(full_image_path)
            ax.imshow(image)
            ax.set_title(f"{image_path}")
            ax.axis("off")

        for ax in axes[len(smallest_k_df) :]:
            ax.axis("off")

        if plot_base_path:
            plot_file_path = os.path.join(
                plot_base_path, f"smallest_{smallest_k}_{metric_name}"
            )
            if filter_label is not None:
                plot_file_path += f"_{filter_label.name.lower()}"
            plot_file_path += ".png"

            plt.savefig(plot_file_path, format="png", dpi=300)
            print(f"Plot saved to {plot_file_path}")

        plt.show()


class InfluenceEvaluator:
    def __init__(
        self,
        metrics: Optional[List[EvalMetric]] = None,
    ):
        if metrics is None:
            metrics = [metric.value for metric in DefaultMetrics]

        self.metrics = metrics

    @property
    def metric_names(self) -> List[str]:
        return [metric.name for metric in self.metrics]

    def get_eval_result(
        self, eval_input: InfluenceEvaluationInput
    ) -> InfluenceEvaluationResult:
        df = self._generate_df(eval_input)
        metric_columns = self.metric_names
        return InfluenceEvaluationResult(df, metric_columns)

    def _generate_df(self, eval_input: InfluenceEvaluationInput) -> pd.DataFrame:
        df = pd.DataFrame(index=eval_input.train_indices)
        df[COL_FILE_PATH] = eval_input.train_relative_file_path
        df[COL_LABEL] = eval_input.train_labels
        df[COL_LABEL_NAME] = [
            Label(label).name.lower() for label in eval_input.train_labels
        ]

        for metric in self.metrics:
            df[metric.name] = metric.compute(eval_input)

        return df
