import glob
import os
from typing import Optional

from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS


class ImageDataset:
    def __init__(
        self, root_folder: Optional[str] = None, files: Optional[list[str]] = None
    ) -> None:
        self.images = []
        if root_folder is not None:
            file_patterns = [f"*{ext}" for ext in IMG_EXTENSIONS]
            for file_pattern in file_patterns:
                self.images.extend(
                    glob.glob(
                        os.path.join(root_folder, "**", file_pattern), recursive=True
                    )
                )
        elif files is not None:
            self.images = files
        else:
            raise RuntimeError("One of root_folder and files should not be None.")

    def __getitem__(self, index: int) -> tuple[Image.Image, str, str]:
        image_path = self.images[index]
        with Image.open(image_path) as f:
            image = f.copy()
            content_type = f.get_format_mimetype()

        return image, image_path, content_type

    def __len__(self) -> int:
        return len(self.images)
