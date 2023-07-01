import glob
import os
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import ToTensor


class ImageDataset:
    def __init__(self, root_folder: str) -> None:
        self.images = []
        file_patterns = [f"*{ext}" for ext in IMG_EXTENSIONS]
        for file_pattern in file_patterns:
            self.images.extend(
                glob.glob(os.path.join(root_folder, "**", file_pattern), recursive=True)
            )

    def __getitem__(self, index: int) -> tuple[np.ndarray, str, str]:
        image_path = self.images[index]
        with Image.open(image_path) as f:
            image = np.array(f)
            content_type = f.get_format_mimetype()

        return image, image_path, content_type

    def __len__(self) -> int:
        return len(self.images)


class TorchImageDataset(TorchDataset[tuple[Optional[torch.Tensor], str, str]]):
    def __init__(
        self,
        root_folder: str,
        transform: Callable = ToTensor(),
        load_image: bool = True,
    ) -> None:
        self.load_image = load_image
        self.transform = transform

        self.images = []
        file_patterns = [f"*{ext}" for ext in IMG_EXTENSIONS]
        for file_pattern in file_patterns:
            self.images.extend(
                glob.glob(os.path.join(root_folder, "**", file_pattern), recursive=True)
            )

    def __getitem__(self, index: int) -> tuple[Optional[torch.Tensor], int, str, str]:
        image_path = self.images[index]

        image = None
        with Image.open(image_path) as f:
            content_type = f.get_format_mimetype()
            if self.load_image:
                image = self.transform(f)

        return image, image_path, content_type

    def __len__(self) -> int:
        return len(self.images)
