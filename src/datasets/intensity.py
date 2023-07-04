

from .weather import WeatherImagesDataset

INTENSITY_CLASSES = ["dark", "medium", "bright"]
INTENSITY_CLASS_TO_ID = {class_: id for id, class_ in enumerate(INTENSITY_CLASSES)}


class IntensityImagesDataset(WeatherImagesDataset):
    pass
