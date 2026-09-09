from typing import Tuple

import numpy as np

from .base_classes import Setup
from .data_importer import PetData
from .mbn_statistics import (
    compute_network_weights,
    multiple_comparison_correction,
    threshold_correction,
)


def compute_conventional(data: PetData, setup: Setup) -> Tuple[np.ndarray, np.ndarray]:
    n_vois = data[0][0].shape[1]
    n_groups = len(data)

    weights_corrected = np.zeros((n_vois, n_vois, n_groups))
    pval_corrected = np.zeros((n_vois, n_vois, n_groups))

    for n, data_tuple in enumerate(data):
        data_array = data_tuple[0]
        weights_corrected[:, :, n], pval_corrected[:, :, n] = build_convetional_mbn(
            data_array, setup
        )

    return weights_corrected, pval_corrected


def build_convetional_mbn(array: np.ndarray, setup: Setup) -> Tuple[np.ndarray, np.ndarray]:
    weights, pvalue = compute_network_weights(array, setup)
    weights_corrected, pvalue_corrected = multiple_comparison_correction(weights, pvalue, setup)
    weights_corrected, pvalue_corrected = threshold_correction(
        weights_corrected, pvalue_corrected, setup
    )
    return weights_corrected, pvalue_corrected
