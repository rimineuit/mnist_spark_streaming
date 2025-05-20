from typing import List
import numpy as np

class Normalize:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image / 255.0
        image = (image - self.mean) / self.std
        return image

class Transforms:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def transform(self, image: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image