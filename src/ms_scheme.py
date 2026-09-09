import logging
from typing import Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .base_classes import MSComputations, Setup
from .data_importer import PetData
from .mbn_statistics import (
    compute_network_weights,
    multiple_comparison_correction,
    threshold_correction,
)


def compute_ms(data: PetData, setup: Setup) -> MSComputations:
    (
        weights_corrected,
        pval_corrected,
        weights_noncorrected,
        pval_noncorrected,
        prob_mat,
    ) = multiple_sampling_scheme(data, setup)

    n_classes = len(data)
    # Eq. 5: retain an edge only if its probability of occurrence exceeds theta
    binary_map = prob_mat > setup.theta

    # Apply threshold with probability map to all generated (corrected) matrices
    for n in range(n_classes):
        mask = binary_map[:, :, n].ravel()
        weights_corrected[~mask, :, n] = 0
        pval_corrected[~mask, :, n] = 1

    return MSComputations(weights_corrected, pval_corrected, weights_noncorrected, pval_noncorrected)


def multiple_sampling_scheme(
    data: PetData, setup: Setup
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_classes = len(data)
    n_workers = int(getattr(setup, "n_workers", 1))

    # One seed per group drawn from the global RNG, so results are reproducible
    # regardless of whether groups are processed sequentially or in parallel workers.
    seeds = np.random.randint(0, 2**31 - 1, size=n_classes)

    output = Parallel(n_jobs=n_workers)(
        delayed(sample_data)(data[c], setup, seed=int(seeds[c])) for c in tqdm(range(n_classes))
    )

    weights_corrected, pval_corrected, weights_noncorrected, pval_noncorrected, prob_mat = zip(*output)
    return (
        np.stack(weights_corrected, axis=-1),
        np.stack(pval_corrected, axis=-1),
        np.stack(weights_noncorrected, axis=-1),
        np.stack(pval_noncorrected, axis=-1),
        np.stack(prob_mat, axis=-1),
    )


def sample_data(
    data_group: tuple, setup: Setup, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate setup.n_samples networks for one group (Algorithm 1, lines 2-6, plus the Pmap of Eq. 3).

    Returns flattened (n_vois**2, n_samples) arrays of corrected/non-corrected weights and
    p-values, and the (n_vois, n_vois) probability map.
    """
    if seed is not None:
        np.random.seed(seed)

    n_vois = data_group[0].shape[1]

    acc_weights_corrected = np.zeros((n_vois**2, setup.n_samples))
    acc_pval_corrected = np.zeros((n_vois**2, setup.n_samples))
    acc_weights_noncorrected = np.zeros((n_vois**2, setup.n_samples))
    acc_pval_noncorrected = np.zeros((n_vois**2, setup.n_samples))
    prob_mat = np.zeros((n_vois, n_vois))

    n_rows = data_group[0].shape[0]

    logging.info(f">> Generating {setup.n_samples} {setup.random_type} samples for group {data_group[1]}")
    # Generate multiple networks
    for k in tqdm(range(setup.n_samples)):
        new_rows = gen_rows(setup, n_rows)
        gen_data = data_group[0][new_rows, :]

        # Compute network weights (e.g Pearson correlation), Eq. 1
        weights, pvalue = compute_network_weights(gen_data, setup)

        # Apply multiple comparison correction (e.g. fdr, bonferroni)
        weights_corrected, pvalue_corrected = multiple_comparison_correction(weights, pvalue, setup)

        # Apply the absolute weight threshold defined in setup.threshold
        weights_corrected, pvalue_corrected = threshold_correction(weights_corrected, pvalue_corrected, setup)

        # Accumulate corrected weights for each sample
        acc_weights_corrected[:, k] = weights_corrected.ravel()
        acc_pval_corrected[:, k] = pvalue_corrected.ravel()

        # Accumulate non corrected weights (raw M^k used by the representative criteria)
        acc_weights_noncorrected[:, k] = weights.ravel()
        acc_pval_noncorrected[:, k] = pvalue.ravel()

        # Accumulate binary maps, Eq. 3
        prob_mat += (weights_corrected != 0).astype("uint8")

    # Normalize prob_mat
    prob_mat = prob_mat / setup.n_samples
    return (
        acc_weights_corrected,
        acc_pval_corrected,
        acc_weights_noncorrected,
        acc_pval_noncorrected,
        prob_mat,
    )


def gen_rows(setup: Setup, n_rows: int) -> np.ndarray:
    if setup.random_type == "bootstrap":
        # Bootstrap Indices (i.e. random select repeated indices within some interval)
        new_rows = np.random.randint(n_rows, size=n_rows)
    elif setup.random_type == "subsampling":
        min_remov = getattr(setup, "min_remov", 0.005)
        max_remov = getattr(setup, "max_remov", 0.10)
        # generate a random removal proportion between [min_remov, max_remov]
        prop = min_remov + (max_remov - min_remov) * np.random.rand()
        # generate the number of rows to remove
        n_remove = round(prop * n_rows)
        # generate the rows that will be used for sample the new data
        new_rows = np.random.permutation(n_rows)[: n_rows - n_remove]
    else:
        raise ValueError(f"Unknown random_type: {setup.random_type!r} (expected 'bootstrap' or 'subsampling')")
    return new_rows


def count_connections(data_matrix):
    row, _ = np.shape(data_matrix)
    connections = np.zeros(row)
    vicinity_graph = [[] for _ in range(row)]

    for u in range(row):
        connections[u] = np.count_nonzero(data_matrix[u, :]) - 1

        idc = np.where(data_matrix[u, :])[0]
        logic_vec = idc != u
        vicinity_graph[u] = idc[logic_vec].tolist()

    counts = connections
    return counts, vicinity_graph, data_matrix


def find_connecting_edges(vicinity_graph):
    n_nodes = len(vicinity_graph)
    edges = None

    for i in range(n_nodes):
        edges_i = [[i, j] for j in vicinity_graph[i]]
        edges = edges_i if edges is None else np.concatenate((edges, edges_i))

    edges = np.sort(edges, axis=1)
    _, idx, counts = np.unique(edges, axis=0, return_counts=True, return_index=True)
    edges = edges[np.sort(idx)]

    node_s = edges[:, 0]
    node_t = edges[:, 1]

    return node_s, node_t


def find_bins_degree_distribution(n_nonrep, n_nodes):
    # Allocate memory
    edge_combination1 = np.zeros((n_nonrep, 2))
    edge_combination2 = np.zeros((n_nonrep, 2))

    # Define bins
    cont2 = 0
    for n in range(1, n_nodes + 1):
        cont_aux = n + 1
        while cont_aux <= n_nodes:
            edge_combination1[cont2, :] = [n, cont_aux]
            edge_combination2[cont2, :] = [cont_aux, n]
            cont_aux += 1
            cont2 += 1

    return edge_combination1, edge_combination2
