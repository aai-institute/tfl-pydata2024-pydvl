import logging
import os
from typing import Callable

import zarr
import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.loggers import CSVLogger

from pydvl.influence.torch import DirectInfluence
from pydvl.influence.torch.util import TorchNumpyConverter
from pydvl.influence import SequentialInfluenceCalculator

from config import (
    data_path,
    result_path,
    data_cell_path,
)
from malaria.dataset import MalariaKaggleDataset
from malaria.eval.evaluator import InfluenceEvaluationInput, InfluenceEvaluator
from malaria.model import LitResnet18SmallBinary
from malaria.train_util import simple_train, model_tag_builder

# Set random seed for reproducibility
RANDOM_SEED = 31
torch.manual_seed(RANDOM_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

# Construct torch dataset objects from Malaria dataset
torch_dataset = MalariaKaggleDataset(data_path).get_torch_dataset()
train_size = int(0.7 * len(torch_dataset))
val_size = int(0.15 * len(torch_dataset))
test_size = len(torch_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(
    torch_dataset,
    [train_size, val_size, test_size],
    torch.Generator().manual_seed(RANDOM_SEED),
)

# Result base directory based on seed and split parameters
result_base_dir_seed = os.path.join(
    result_path, f"seed={RANDOM_SEED}-sizes={train_size}_{val_size}_{test_size}"
)


def train(epochs: int, b_size: int):
    logging_path = os.path.join(result_path, "logs")
    logger = CSVLogger(logging_path, name="malaria_lightning")
    checkpoint_dir = os.path.join(
        result_base_dir_seed,
        "model_checkpoints",
    )
    model = LitResnet18SmallBinary()

    model = simple_train(
        model,
        train_dataset,
        val_dataset,
        b_size,
        epochs,
        checkpoint_dir,
        logger=logger,
    )
    return model


def compute_influences(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model_identifier: str,
    load: bool = True,
) -> NDArray:
    test_train_value_path = os.path.join(
        result_base_dir_seed, "influences", model_identifier
    )

    if os.path.exists(test_train_value_path) and load:
        z = zarr.open(test_train_value_path, mode="r")
        return z[:]

    test_batch_size = 256
    train_batch_size = 256
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    model.eval()
    if_model = DirectInfluence(model, loss_fn, hessian_regularization=0.001)
    if_model.fit(train_loader)

    if_calc = SequentialInfluenceCalculator(if_model)
    lazy_if_test_train_values = if_calc.influences(test_loader, train_loader)

    test_train_values = lazy_if_test_train_values.to_zarr(
        test_train_value_path, TorchNumpyConverter(), overwrite=True, return_stored=True
    )
    return test_train_values[:]


def generate_plots(test_train_values: NDArray, model_identifier: str, load: bool = True):

    eval_input_base_dir = os.path.join(result_base_dir_seed, "eval_input")
    test_train_eval_input_path = os.path.join(
        eval_input_base_dir, f"{model_identifier}.pkl"
    )

    if os.path.exists(test_train_eval_input_path) and load:
        test_train_eval_input = InfluenceEvaluationInput.from_pickle(test_train_eval_input_path)
    else:
        test_train_eval_input = InfluenceEvaluationInput(
            np.asarray(test_train_values),
            torch_dataset,
            test_dataset.indices,
            train_dataset.indices,
        )
        os.makedirs(eval_input_base_dir, exist_ok=True)
        test_train_eval_input.to_pickle(test_train_eval_input_path)

    evaluator = InfluenceEvaluator()
    eval_result = evaluator.get_eval_result(test_train_eval_input)

    hist_kwargs = {"bins": 10,
                   "element": "step",
                   #"log_scale": (False, True),
                   "common_norm": False,
                   "stat": "density"
                   }

    metric_name = "mean_influence_uninfected"

    plot_dir = os.path.join(result_base_dir_seed, "plots", model_tag)
    eval_result.generate_hist_plot(metric_name,
                                   plot_base_path=plot_dir,
                                   parasitized_only=True, **hist_kwargs)

    eval_result.generate_smallest_k_by_metric_quantile(metric_name,
                                                       0.25,
                                                       3,
                                                       data_cell_path,
                                                       parasitized_only=True,
                                                       plot_base_path=plot_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    batch_size = 768
    max_epochs = 500
    model_tag = model_tag_builder(max_epochs, batch_size)

    lit_model = train(max_epochs, batch_size)
    torch_model = lit_model.model
    loss = lit_model.loss_fn

    influence_array = compute_influences(torch_model, loss, model_tag)
    generate_plots(influence_array, model_tag)

