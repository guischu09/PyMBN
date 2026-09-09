"""Regression tests for the MS scheme against the method described in the paper."""
import dataclasses
import os
import pickle

import numpy as np
from scipy.special import logsumexp
from statsmodels.stats.multitest import multipletests

from src.base_classes import MSComputations
from src.mbn_builder import compute_representative_mbn
from src.mbn_statistics import multiple_comparison_correction, pearson_correlation, threshold_correction
from src.ms_scheme import compute_ms, gen_rows, sample_data
from src.ui_parser import ManualSetup


def load_demo_data():
    path = os.path.join(os.getcwd(), "tests", "data", "pet_data_demo.pickle")
    with open(path, "rb") as f:
        return pickle.load(f)


def small_setup(**overrides):
    setup = ManualSetup().get_parameters()
    setup.n_samples = 20
    return dataclasses.replace(setup, **overrides)


def test_fdr_uses_upper_triangle_only():
    class S:
        alpha = 0.05
        correction_type = "fdr_bh"

    rng = np.random.default_rng(0)
    X = rng.normal(size=(25, 8))
    X[:, 1] += 2 * X[:, 0]
    w, p = pearson_correlation(X)
    _, q = multiple_comparison_correction(w, p, S)

    iu = np.triu_indices(8, k=1)
    expected = multipletests(p[iu], method="fdr_bh")[1]
    assert np.allclose(q[iu], expected)
    assert np.allclose(q, q.T)
    assert np.all(np.diag(q) == 0)


def test_threshold_keeps_strong_negative_edges_and_does_not_mutate():
    class S:
        threshold = 0.3

    w = np.array([[1.0, -0.9, 0.1], [-0.9, 1.0, 0.5], [0.1, 0.5, 1.0]])
    p = np.zeros_like(w)
    w_in, p_in = w.copy(), p.copy()
    wc, pc = threshold_correction(w, p, S)
    assert wc[0, 1] == -0.9 and wc[0, 2] == 0 and wc[1, 2] == 0.5
    assert pc[0, 2] == 1
    assert np.array_equal(w, w_in) and np.array_equal(p, p_in)


def test_sample_data_applies_fdr_and_keeps_raw_networks():
    setup = small_setup()
    data = load_demo_data()
    n_vois = data[0][0].shape[1]
    wc, pc, wn, pn, prob = sample_data(data[0], setup, seed=1)

    # raw networks are untouched: no zeros off the diagonal, p-values from the t-test
    off_diag = ~np.eye(n_vois, dtype=bool).ravel()
    assert np.all(wn[off_diag, :] != 0)
    assert not np.array_equal(wc, wn)

    # every retained corrected edge is FDR-significant and above the absolute threshold
    for k in range(setup.n_samples):
        w = wn[:, k].reshape(n_vois, n_vois)
        p = pn[:, k].reshape(n_vois, n_vois)
        expected, _ = multiple_comparison_correction(w, p, setup)
        expected, _ = threshold_correction(expected, np.ones_like(p), setup)
        assert np.allclose(wc[:, k].reshape(n_vois, n_vois), expected)

    # probability map is the fraction of samples in which each edge survived (Eq. 3)
    assert np.allclose(prob, np.mean(wc != 0, axis=1).reshape(n_vois, n_vois))


def test_pmap_threshold_is_strict():
    setup = small_setup(theta=0.5)
    data = load_demo_data()
    ms = compute_ms(data, setup)
    n_vois = data[0][0].shape[1]
    for g in range(len(data)):
        prob = np.mean(ms.weights_corrected[:, :, g] != 0, axis=1).reshape(n_vois, n_vois)
        # after masking, every surviving edge had probability strictly above theta,
        # and edges at or below theta are gone
        retained = np.any(ms.weights_corrected[:, :, g] != 0, axis=1).reshape(n_vois, n_vois)
        assert np.all(prob[retained] > setup.theta)


def test_seeded_results_are_reproducible():
    data = load_demo_data()
    out1 = sample_data(data[0], small_setup(), seed=7)
    out2 = sample_data(data[0], small_setup(), seed=7)
    for a, b in zip(out1, out2):
        assert np.array_equal(a, b)


def test_subsampling_generates_valid_rows():
    setup = small_setup(random_type="subsampling")
    np.random.seed(0)
    for _ in range(50):
        rows = gen_rows(setup, 100)
        assert len(np.unique(rows)) == len(rows)
        assert 100 - round(setup.max_remov * 100) <= len(rows) <= 100 - round(setup.min_remov * 100)


def _fake_ms(n_vois=4, n_samples=15, n_class=2, seed=3):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_vois**2, n_samples, n_class))
    corrected = raw * (rng.random(raw.shape) > 0.3)
    return MSComputations(corrected, np.zeros_like(raw), raw, np.zeros_like(raw))


def test_mean_median_mode_match_reference_implementations():
    ms = _fake_ms()
    n_vois = 4
    for crit, expected_idx in [
        ("mean", lambda s: np.argmin(np.linalg.norm(s - s.mean(1, keepdims=True), axis=0))),
        ("median", lambda s: np.argmin(np.linalg.norm(s - np.median(s, 1, keepdims=True), axis=0))),
        ("mode", lambda s: np.argmax([logsumexp(-0.5 * np.sum((s - s[:, [k]]) ** 2, axis=0)) for k in range(s.shape[1])])),
    ]:
        setup = small_setup(criteria_representation=crit)
        w, _ = compute_representative_mbn(ms, setup)
        for g in range(2):
            idx = expected_idx(ms.weights_noncorrected[:, :, g])
            assert np.array_equal(w[:, :, g], ms.weights_corrected[:, idx, g].reshape(n_vois, n_vois)), crit


def test_geodesic_criterion_runs():
    ms = _fake_ms()
    w, _ = compute_representative_mbn(ms, small_setup(criteria_representation="geodesic"))
    assert w.shape == (4, 4, 2)
