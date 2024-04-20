import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from kaggle import KaggleApi


def load_via_kaggle(dataset_name: str, destination_path: str, force: bool = False):
    # Authenticating with the Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Downloading the dataset
    api.dataset_download_files(
        dataset_name, path=destination_path, unzip=True, force=force, quiet=False
    )


class MalariaKaggleDataset:
    def __init__(self, data_base_dir: str):
        self.data_base_dir = data_base_dir
        self.kaggle_dataset_name = "iarunava/cell-images-for-detecting-malaria"
        self.file_name = "abc"
        self.parasite_path = os.path.join(
            self.data_base_dir, "cell_images", "Parasitized"
        )
        self.uninfected_path = os.path.join(
            self.data_base_dir, "cell_images", "Uninfected"
        )
        self.file_path = os.path.join(self.data_base_dir, self.file_name)

    def download(self, reload: bool = False):
        if not self._data_exists() or reload:
            try:
                load_via_kaggle(
                    self.kaggle_dataset_name, self.data_base_dir, force=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download dataset "
                    f"{self.kaggle_dataset_name} to {self.file_path}"
                    f"via Kaggle API. Consider to download the data"
                    f"manually: {e}"
                )

    def _data_exists(self) -> bool:
        parasite_is_empty = os.listdir(self.parasite_path) == 0
        uninfected_is_empty = os.listdir(self.uninfected_path) == 0
        any_is_empty = parasite_is_empty or uninfected_is_empty
        return not any_is_empty

    def get_torch_dataset(self, reload: bool = False) -> ImageFolder:
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
        return ImageFolder(
            os.path.join(self.data_base_dir, "cell_images"), transform=transform
        )
