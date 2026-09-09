from typing import Optional, Tuple

import numpy as np
from scipy import linalg
from scipy.special import logsumexp

from .base_classes import MSComputations, Setup
from .conventional_method import compute_conventional
from .data_importer import PetData
from .ms_scheme import compute_ms


def build_network(data: PetData, setup: Setup) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build the group representative networks.

    Returns the (n_vois, n_vois, n_groups) representative weights and, for the MS scheme,
    the (n_vois**2, n_samples, n_groups) collection of corrected sampled networks
    (None for the conventional method).
    """
    if setup.mbn_method == "ms_scheme":
        ms_data = compute_ms(data, setup)
        group_networks, _ = compute_representative_mbn(ms_data, setup)
        return group_networks, ms_data.weights_corrected

    if setup.mbn_method == "conventional":
        group_networks, _ = compute_conventional(data, setup)
        return group_networks, None

    raise ValueError(f"Unknown mbn_method: {setup.mbn_method!r} (expected 'ms_scheme' or 'conventional')")


def compute_representative_mbn(ms_data: MSComputations, setup: Setup) -> Tuple[np.ndarray, np.ndarray]:
    OPTIONS = {"mean": mean_mbn, "median": median_mbn, "mode": mode_mbn, "geodesic": geodesic_mbn}

    if setup.criteria_representation not in OPTIONS:
        raise ValueError(
            f"Unknown criteria_representation: {setup.criteria_representation!r} "
            f"(expected one of {sorted(OPTIONS)})"
        )
    process_fc = OPTIONS[setup.criteria_representation]

    data = process_fc(ms_data)
    return data


def _select_representatives(ms: MSComputations, corr_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pick, for each group, the FDR/Pmap corrected network of the selected sample index."""
    n_row, _, n_class = np.shape(ms.weights_corrected)
    dim = int(np.sqrt(n_row))
    weights_representative = np.zeros((dim, dim, n_class))
    pval_representative = np.zeros((dim, dim, n_class))

    for u in range(n_class):
        weights_representative[:, :, u] = ms.weights_corrected[:, corr_index[u], u].reshape(dim, dim)
        pval_representative[:, :, u] = ms.pval_corrected[:, corr_index[u], u].reshape(dim, dim)

    return weights_representative, pval_representative


def _closest_to_center(ms: MSComputations, center_fn) -> np.ndarray:
    """Index of the sampled (raw) network closest in Frobenius norm to a central matrix (Eq. 2)."""
    _, _, n_class = np.shape(ms.weights_noncorrected)
    corr_index = np.zeros(n_class, dtype=int)

    for u in range(n_class):
        samples = ms.weights_noncorrected[:, :, u]  # (n_vois**2, n_samples)
        center = center_fn(samples, axis=1)
        dist = np.sqrt(np.sum((samples - center[:, None]) ** 2, axis=0))
        corr_index[u] = np.argmin(dist)

    return corr_index


def mean_mbn(ms: MSComputations) -> Tuple[np.ndarray, np.ndarray]:
    """Mean criterion: the sample matrix M^k that best approximates the mean matrix (Eq. 2)."""
    corr_index = _closest_to_center(ms, np.nanmean)
    return _select_representatives(ms, corr_index)


def median_mbn(ms: MSComputations) -> Tuple[np.ndarray, np.ndarray]:
    """Median criterion: the sample matrix M^k that best approximates the element-wise median matrix."""
    corr_index = _closest_to_center(ms, np.nanmedian)
    return _select_representatives(ms, corr_index)


def mode_mbn(ms: MSComputations, chunk_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Mode criterion (Supporting Information, Eqs. S1-S4).

    The posterior p(M^k | Phi) is a mixture of isotropic Gaussians (U x V = I) centered
    at every sampled network with uniform weights Pi_t = 1/n. The representative is the
    maximum a posteriori sample, i.e. the one maximizing sum_t exp(-0.5 ||M^k - M^t||_F^2).
    The sum is evaluated in log-space (logsumexp) in chunks to bound memory.
    """
    _, n_samples, n_class = np.shape(ms.weights_noncorrected)
    corr_index = np.zeros(n_class, dtype=int)

    for u in range(n_class):
        vecs = np.ascontiguousarray(ms.weights_noncorrected[:, :, u].T)  # (n_samples, n_vois**2)
        sq_norms = np.sum(vecs**2, axis=1)
        log_post = np.empty(n_samples)

        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            # squared Frobenius distances between the chunk and every sample
            sq_dist = sq_norms[start:stop, None] + sq_norms[None, :] - 2.0 * (vecs[start:stop] @ vecs.T)
            np.maximum(sq_dist, 0.0, out=sq_dist)
            log_post[start:stop] = logsumexp(-0.5 * sq_dist, axis=1)

        corr_index[u] = np.argmax(log_post)

    return _select_representatives(ms, corr_index)


def geodesic_mbn(ms: MSComputations) -> Tuple[np.ndarray, np.ndarray]:
    """Experimental: the sample closest to the mean matrix under the affine-invariant (geodesic)
    distance between symmetric positive definite matrices. Not part of the published method."""
    n_row, n_samples, n_class = np.shape(ms.weights_noncorrected)
    dim = int(np.sqrt(n_row))
    corr_index = np.zeros(n_class, dtype=int)

    for u in range(n_class):
        real_mean = np.nanmean(ms.weights_noncorrected[:, :, u], axis=1).reshape(dim, dim)

        geo_dists = np.zeros(n_samples)
        for vv in range(n_samples):
            Q1 = ms.weights_noncorrected[:, vv, u].reshape(dim, dim)

            # Regularize matrices equally so both are SPD
            reg_mats, _ = regularize_matrices([real_mean, Q1])
            geo_dists[vv] = f_dist_geodesic(reg_mats[0], reg_mats[1])

        corr_index[u] = np.argmin(geo_dists)

    return _select_representatives(ms, corr_index)


def _is_spd(mat: np.ndarray, eps: float = 1e-6) -> bool:
    """True if the (symmetric) matrix is positive definite with a numerical margin."""
    return bool(np.linalg.eigvalsh(mat).min() > eps)


def regularize_matrices(mats_list, step: float = 0.1):
    """Add the same tau * I to every matrix until all of them are symmetric positive definite."""
    dim = mats_list[0].shape[0]
    mats_list = [0.5 * (m + m.T) for m in mats_list]  # enforce symmetry
    tau = 0.0

    while not all(_is_spd(m + tau * np.eye(dim)) for m in mats_list):
        tau += step

    reg_mats_cell = [m + tau * np.eye(dim) for m in mats_list]
    return reg_mats_cell, tau


def f_dist_geodesic(Q1, Q2):
    """Affine-invariant geodesic distance between two SPD matrices:
    sqrt(sum(log(lambda_i)^2)) with lambda_i the generalized eigenvalues of (Q2, Q1)."""
    e = linalg.eigh(Q2, Q1, eigvals_only=True)
    dg = np.sqrt(np.sum(np.log(e) ** 2))

    return dg
