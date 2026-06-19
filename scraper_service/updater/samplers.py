import numpy as np
from abc import ABC, abstractmethod


class RandomSampler(ABC):
    """Sampler base que devuelve muestras numéricas desde una distribución.

    Método `sample(size=1)` devuelve un numpy array de longitud `size`.
    Si `positive_values_only=True`, las muestras negativas se transforman a valores positivos.
    """

    def __init__(self, positive_values_only: bool = True):
        self.positive_values_only = bool(positive_values_only)

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        raise NotImplementedError()


class PoissonSampler(RandomSampler):
    def __init__(self, lam: float, positive_values_only: bool = True):
        super().__init__(positive_values_only=positive_values_only)
        self.lam = float(lam)

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.poisson(self.lam, size=size)
        if self.positive_values_only:
            # asegurar al menos 1 para conteos
            samples = np.clip(samples, 1, None)
        return samples


class NormalSampler(RandomSampler):
    def __init__(self, mean: float, std: float, positive_values_only: bool = True):
        super().__init__(positive_values_only=positive_values_only)
        self.mean = float(mean)
        self.std = float(std)

    def sample(self, size: int = 1) -> np.ndarray:
        samples = np.random.normal(self.mean, self.std, size=size)
        if self.positive_values_only:
            # transformar negativos a su valor absoluto para mantener magnitud
            samples = np.where(samples <= 0, np.abs(samples) + 1e-6, samples)
        return samples