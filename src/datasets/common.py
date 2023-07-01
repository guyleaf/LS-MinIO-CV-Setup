import glob
import os

import numpy as np
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS


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
