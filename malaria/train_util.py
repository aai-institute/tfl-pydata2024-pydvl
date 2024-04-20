import os
from typing import Optional

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import Logger


def model_tag_builder(max_epochs: int, batch_size: int) -> str:
    return f"{max_epochs=}-{batch_size=}"


def simple_train(
    model: LightningModule,
    train_data: Dataset,
    val_data: Dataset,
    batch_size: int,
    max_epochs: int,
    checkpoint_base_dir: str,
    logger: Optional[Logger] = None,
    load: bool = True,
) -> LightningModule:
    """
    Simple training routine

    Args:
        logger:
        model:
        train_data:
        val_data:
        batch_size:
        max_epochs:
        checkpoint_base_dir:
        load:

    Returns:

    """

    file_name = model_tag_builder(max_epochs, batch_size)

    checkpoint_path = f"{os.path.join(checkpoint_base_dir, file_name)}.ckpt"

    if os.path.exists(checkpoint_path) and load:
        return type(model).load_from_checkpoint(checkpoint_path)

    model_checkpoint = ModelCheckpoint(
        checkpoint_base_dir,
        filename=file_name,
        save_top_k=1,
        enable_version_counter=False,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=28
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=5,
        logger=logger,
        callbacks=[model_checkpoint],
    )
    trainer.fit(model, train_loader, val_loader)

    best_model_path = model_checkpoint.best_model_path
    return type(model).load_from_checkpoint(best_model_path)
