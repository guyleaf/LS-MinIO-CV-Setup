import numpy as np
import skimage.color as skcolor

from ..datasets.common import ImageDataset
from ..datasets.intensity import INTENSITY_CLASS_TO_ID, INTENSITY_CLASSES


def measure_intensity_level(
    root_dir: str, thresholds: list[float] = [0.33, 0.67]
) -> np.ndarray:
    if len(thresholds) != 2:
        raise ValueError("The thresholds should only contain two floating values")

    levels = []
    dataset = ImageDataset(root_dir)
    for rgb, _, _ in dataset:
        lab = skcolor.rgb2lab(rgb)
        intensity = lab[..., 0] / 100
        average_intensity = np.mean(intensity)

        level = "medium"
        if average_intensity >= thresholds[1]:
            level = "bright"
        elif average_intensity <= thresholds[0]:
            level = "dark"
        levels.append(INTENSITY_CLASS_TO_ID[level])

    return np.array(levels)


def collect_intensity_level_from_tasks(
    tasks: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    intensities_1 = []
    intensities_2 = []
    for task in tasks:
        for annotation in task["annotations"][0]["result"]:
            if annotation["from_name"] == "intensity_1":
                level = annotation["value"]["choices"][0]
                intensities_1.append(INTENSITY_CLASS_TO_ID[level])
            elif annotation["from_name"] == "intensity_2":
                level = annotation["value"]["choices"][0]
                intensities_2.append(INTENSITY_CLASS_TO_ID[level])

    return np.array(intensities_1), np.array(intensities_2)


def get_intensity_classes() -> list[str]:
    return INTENSITY_CLASSES
