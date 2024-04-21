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
)
from malaria.dataset import MalariaKaggleDataset, Label
from malaria.eval.evaluator import InfluenceEvaluationInput, InfluenceEvaluator
from malaria.eval.metrics import DefaultMetrics
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


def train(epochs: int, b_size: int, load: bool = True):
    """
    A closure (train_dataset, val_dataset, result_path) for training a simple Resnet18
    pytorch lightning model for binary classification
    Args:
        epochs: max number of epochs
        b_size: batch size for training and validation
        load: whether to load a previously persisted result;
            if False, do not load an old result but store the newly computed result

    Returns:
        The trained lightning model from the epoch with the smallest validation loss
    """
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
        load=load
    )
    return model


def compute_influences(
    model: torch.nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model_identifier: str,
    chunk_size: int,
    load: bool = True,
) -> NDArray:
    """
    A closure (train_dataset, test_dataset, result_base_dir_seed) for computing
    influence values between train and test data given a trained model and loss
    function.

    Args:
        model: The Hessian will be calculated with respect to
            this model's parameters.
        loss_fn: A callable that takes the model's output and target as input and returns
              the scalar loss.
        model_identifier: a unique identifier for the model; used for cacheing
        chunk_size: determines the chunking of the test and train data; if you observe
            memory issues, reduce this numer
        load: whether to load a previously persisted result; if False, do not load an
            old result but store the newly computed result

    Returns:
        A numpy array containing the computed influences
    """

    test_train_value_path = os.path.join(
        result_base_dir_seed, "influences", model_identifier
    )

    # retrieve cached values, if available
    if os.path.exists(test_train_value_path) and load:
        z = zarr.open(test_train_value_path, mode="r")
        return z[:]

    # define loaders based on the desired chunk size
    test_loader = DataLoader(test_dataset, batch_size=chunk_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=chunk_size, shuffle=False)

    # turn torch model into eval mode and fit the corresponding influence function model
    # to the train data
    model.eval()
    if_model = DirectInfluence(model, loss_fn, hessian_regularization=0.001)
    if_model.fit(train_loader)

    # Wrap the chunk computation model into a sequential calculator. No computation is
    # triggered, when calling the influence method
    if_calc = SequentialInfluenceCalculator(if_model)
    lazy_if_test_train_values = if_calc.influences(test_loader, train_loader)

    # Trigger computation and write results chunk-wise to disk
    test_train_values = lazy_if_test_train_values.to_zarr(
        test_train_value_path, TorchNumpyConverter(), overwrite=True, return_stored=True
    )
    return test_train_values[:]


def generate_plots(
    test_train_values: NDArray, model_identifier: str, load: bool = True
):
    """
    A closure (torch_dataset, train_dataset, test_dataset, result_base_dir_seed) to
    generate histograms and plot negative influence values
    Args:
        test_train_values: precomputed influence values
        model_identifier: a unique identifier for the model; used for cacheing
        load: whether to load a previously persisted result; if False, do not load an
            old result but store the newly computed result

    """
    eval_input_base_dir = os.path.join(result_base_dir_seed, "eval_input")
    test_train_eval_input_path = os.path.join(
        eval_input_base_dir, f"{model_identifier}.pkl"
    )

    if os.path.exists(test_train_eval_input_path) and load:
        test_train_eval_input = InfluenceEvaluationInput.from_pickle(
            test_train_eval_input_path
        )
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

    hist_kwargs = {
        "bins": 10,
        "element": "step",
        "common_norm": False,
        "stat": "density",
    }

    metrics_to_plot = [default_metric.value for default_metric in DefaultMetrics]
    filter_labels = [None, Label.UNINFECTED, Label.PARASITIZED]

    plot_dir = os.path.join(result_base_dir_seed, "plots", model_tag)

    for metric in metrics_to_plot:
        for filter_label in filter_labels:
            eval_result.generate_hist_plot(
                metric,
                plot_base_path=plot_dir,
                filter_label=filter_label,
                **hist_kwargs,
            )

            eval_result.generate_smallest_k_by_metric(
                metric,
                3,
                os.path.join(data_path, "cell_images"),
                filter_label=filter_label,
                plot_base_path=plot_dir,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    max_epochs = 500
    # To fit your RAM, you can adapt these parameters
    batch_size = 3072
    influence_chunk_size = 1024

    model_tag = model_tag_builder(max_epochs, batch_size)

    # train a simple torch model
    lit_model = train(max_epochs, batch_size)
    torch_model = lit_model.model
    loss = lit_model.loss_fn

    # compute influences between train and test data
    influence_array = compute_influences(
        torch_model, loss, model_tag, influence_chunk_size
    )

    # generate histograms, extract negative influential data point and plot
    # corresponding images
    generate_plots(influence_array, model_tag)
