import os
from enum import Enum
import shutil

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from kaggle import KaggleApi


def load_via_kaggle(dataset_name: str, destination_path: str, force: bool = False):
    """
    Authenticates with the Kaggle API and downloads a specified dataset.

    This function initializes the Kaggle API, authenticates, and then downloads
    the dataset specified by `dataset_name` to the path given by `destination_path`.
    If `force` is set to True, it will overwrite any existing files in the destination
    directory.

    Args:
        dataset_name: The name of the dataset to download.
            Should be in the format 'user/dataset'.
        destination_path: The local path where the dataset should be saved after
            downloading. If the path does not exist, it will be created.
        force: Optional; If True, existing files at the destination path will be
            overwritten.

    Returns:
        None
    """

    # Authenticating with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Downloading the dataset
    api.dataset_download_files(
        dataset_name, path=destination_path, unzip=True, force=force, quiet=False
    )


class Label(Enum):
    """
    Enumeration for labeling cell images based on their infection status.

    Attributes:
        UNINFECTED: An enum member representing an uninfected status.
        PARASITIZED: An enum member representing a parasitized status.

    """

    UNINFECTED = 1
    PARASITIZED = 0


class MalariaKaggleDataset:
    """
    A class to manage the downloading, organizing, and loading of a malaria dataset
    from Kaggle.

    This class provides functionality to download the malaria cell images dataset
    from Kaggle, check if the data exists locally, and prepare it for use with a
    PyTorch model by applying necessary transformations.

    Attributes:
        data_base_dir (str): The base directory where the dataset will be stored.
        kaggle_dataset_name (str): The name of the dataset on Kaggle.
        cell_images_base_dir (str): The directory where cell images are stored.
        parasite_path (str): Path to the 'Parasitized' images directory.
        uninfected_path (str): Path to the 'Uninfected' images directory.
    """

    def __init__(self, data_base_dir: str):
        """
        Initializes the dataset object with paths and dataset details.

        Args:
            data_base_dir: The base directory for dataset storage.
        """
        self.data_base_dir = data_base_dir
        self.kaggle_dataset_name = "iarunava/cell-images-for-detecting-malaria"
        self.cell_images_base_dir = os.path.join(self.data_base_dir, "cell_images")

        self.parasite_path = os.path.join(self.cell_images_base_dir, "Parasitized")
        self.uninfected_path = os.path.join(self.cell_images_base_dir, "Uninfected")

    def download(self, reload: bool = False):
        """
        Downloads the dataset from Kaggle to the specified directory.

        Args:
            reload: If True, re-downloads the data even if it already exists.
        """
        if not self._data_exists() or reload:
            try:
                load_via_kaggle(
                    self.kaggle_dataset_name, self.data_base_dir, force=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download dataset "
                    f"{self.kaggle_dataset_name} to {self.cell_images_base_dir}"
                    f"via Kaggle API. Consider to download the data"
                    f"manually: {e}"
                )
        # In case the archive from Kaggle contains an additional subdirectory
        # cell_images (so cell_images/cell_images) delete it if the data
        # exists on the top level (in the folders cell_images/Uninfected and
        # cell_images/Parasitized) or try to move the corresponding sub-folders
        # to the top-level
        false_cell_image_root = os.path.join(self.cell_images_base_dir, "cell_images")

        if self._data_exists():
            if os.path.exists(false_cell_image_root):
                shutil.rmtree(false_cell_image_root)
        else:
            try:
                false_root_parasitized = os.path.join(
                    false_cell_image_root, "Parasitized"
                )
                false_root_uninfected = os.path.join(
                    false_cell_image_root, "Uninfected"
                )
                shutil.move(false_root_parasitized, self.parasite_path)
                shutil.move(false_root_uninfected, self.uninfected_path)
                shutil.rmtree(false_cell_image_root)
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Fail to correct downloaded data. "
                    f"Consider to download the data"
                    f"manually: {e}"
                )

    def _data_exists(self) -> bool:
        """
        Checks if the dataset exists and is correctly organized in the filesystem.

        Returns:
            True if the dataset is correctly organized, False otherwise.
        """
        if not os.path.exists(self.data_base_dir):
            return False
        parasite_is_empty = os.listdir(self.parasite_path) == 0
        uninfected_is_empty = os.listdir(self.uninfected_path) == 0
        any_is_empty = parasite_is_empty or uninfected_is_empty
        return not any_is_empty

    def get_torch_dataset(self, reload: bool = False) -> ImageFolder:
        """
        Provides a PyTorch dataset ready for model training.

        Args:
            reload: Whether to re-download and organize the dataset even if it already
            exists locally.

        Returns:
            A PyTorch ImageFolder dataset with appropriate transformations applied.
        """

        if not self._data_exists() or reload:
            self.download(reload=True)

        transform = Compose(
            [
                Resize((224, 224)),  # Resize the image to 224x224
                ToTensor(),  # Convert the image to a tensor
                Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize
            ]
        )
        return ImageFolder(self.cell_images_base_dir, transform=transform)
