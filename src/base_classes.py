from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class InputHelper(ABC):
    @abstractmethod
    def get_data_filepath(self) -> str:
        pass

    @abstractmethod
    def get_coords_filepath(self) -> str:
        pass

    @abstractmethod
    def get_atlas_filepath(self) -> str:
        pass

    @abstractmethod
    def get_labels_filepath(self) -> str:
        pass


@dataclass
class Setup:
    alpha: float  # significance level for the multiple comparison correction
    theta: float  # probability map threshold (paper: theta = 1 - alpha)
    threshold: float  # minimum absolute edge weight retained
    weights: str
    correction_type: str
    criteria_representation: str  # "mean", "median", "mode" or "geodesic"
    data_balance: str
    n_samples_measures: int
    plot_3d: bool
    brain_type: str
    which_plot: str
    output_format: str
    mbn_method: str  # "ms_scheme" or "conventional"
    n_samples: int  # number of sampled networks (n in the paper)
    random_type: str  # "bootstrap" or "subsampling"
    interactive: bool
    seed: int
    # Subsampling scheme only: fraction of subjects removed per sample is drawn
    # uniformly in [min_remov, max_remov] (paper: S_min = 0.5%, S_max = 10%).
    min_remov: float = 0.005
    max_remov: float = 0.10
    n_workers: int = 1


class SetupHelper(ABC):
    @abstractmethod
    def get_parameters(self) -> Setup:
        pass


@dataclass
class MSComputations:
    weights_corrected: np.ndarray
    pval_corrected: np.ndarray
    weights_noncorrected: np.ndarray
    pval_noncorrected: np.ndarray
