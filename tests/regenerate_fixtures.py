"""Regenerate the test fixtures that are derived from the pipeline itself.

Run from the repository root:  python tests/regenerate_fixtures.py

Only fixtures produced by pipeline code are rewritten here. Fixtures that act as
independent references (e.g. the raw Pearson coefficients) are left untouched.
"""
import os
import pickle

import numpy as np
import pandas as pd

from src.conventional_method import compute_conventional
from src.data_importer import process_data
from src.mbn_statistics import multiple_comparison_correction
from src.ms_scheme import sample_data
from src.ui_parser import ManualSetup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "data")
INPUT = os.path.join(ROOT, "input", "dummy_mice_data.csv")


def regenerate_fdr_fixture():
    class S:
        alpha = 0.05
        correction_type = "fdr_bh"

    weights = pd.read_csv(os.path.join(DATA, "weights_data1.csv")).values
    pvalue = pd.read_csv(os.path.join(DATA, "pval_data1.csv")).values
    weights_corrected, qvalue = multiple_comparison_correction(weights, pvalue, S)
    pd.DataFrame(weights_corrected).to_csv(os.path.join(DATA, "weights_corrected_data1.csv"), index=False)
    pd.DataFrame(qvalue).to_csv(os.path.join(DATA, "qval_data1.csv"), index=False)


def regenerate_conventional_fixtures():
    class S:
        correction_type = "fdr_bh"
        data_balance = "imbalanced"
        weights = "pearson_correlation"
        threshold = 0.3
        alpha = 0.05

    data = process_data(INPUT, S)
    for alpha, suffix in [(0.05, ""), (0.01, "_alpha0.01")]:
        S.alpha = alpha
        weights_corrected, _ = compute_conventional(data, S)
        for g in range(2):
            name = f"dummy_r_conventional_group{g + 1}{suffix}.csv"
            pd.DataFrame(weights_corrected[:, :, g]).to_csv(os.path.join(DATA, name), index=False)


def regenerate_sample_data_fixture():
    setup = ManualSetup().get_parameters()
    np.random.seed(setup.seed)
    with open(os.path.join(DATA, "pet_data_demo.pickle"), "rb") as f:
        data = pickle.load(f)
    output = sample_data(data[0], setup)
    with open(os.path.join(DATA, "sample_data_demo.pickle"), "wb") as f:
        pickle.dump(output, f)


if __name__ == "__main__":
    regenerate_fdr_fixture()
    regenerate_conventional_fixtures()
    regenerate_sample_data_fixture()
    print("Fixtures regenerated.")
