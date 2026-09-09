import os
import sys
import tkinter as tk
from tkinter import filedialog

from .base_classes import InputHelper, Setup, SetupHelper
import logging


logging.basicConfig(level=logging.INFO)


class TkInput(InputHelper):
    def __init__(self, setup) -> None:
        self.data_path = self.get_data_filepath()
        self.plot_3d = setup.plot_3d
        if setup.plot_3d:
            self.coords_path = self.get_coords_filepath()
            self.atlas_path = self.get_atlas_filepath()
            self.labels_path = self.get_labels_filepath()
        else:
            self.labels_path = self.get_labels_filepath()

    def get_data_filepath(self) -> str:
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "input"),
            title="Select a file containing your PET measures (e.g. SUV/SUVr)",
            filetypes=(("data files", "*.csv"), ("all files", "*.*")),
        )
        if self.is_valid_input(filepath):
            return filepath

    def get_coords_filepath(self) -> str:
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "input"),
            title="Select a file containing the volumes of interest coordinates",
            filetypes=(("coords files", "*.csv"), ("all files", "*.*")),
        )
        if self.is_valid_input(filepath):
            return filepath

    def get_labels_filepath(self) -> str:
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "input"),
            title="Select a file containing the volumes of interest labels",
            filetypes=(("coords files", "*.csv"), ("all files", "*.*")),
        )
        if self.is_valid_input(filepath):
            return filepath

    def get_atlas_filepath(self) -> str:
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd(), "input"),
            title="Select an atlas structural image file",
            filetypes=(("atlas files", "*.mnc"), ("all files", "*.*")),
        )
        if self.is_valid_input(filepath):
            return filepath

    @staticmethod
    def is_valid_input(filepath: str):
        if not filepath:
            print("Exiting program.")
            sys.exit(0)
        elif os.path.exists(filepath):
            return True
        else:
            raise TypeError(f"File {filepath} does not exist. Please double check if input path is correct.")


class ManualInput(InputHelper):
    def __init__(self, setup) -> None:
        self.data_path = self.get_data_filepath()
        if setup.plot_3d:
            self.coords_path = self.get_coords_filepath()
            self.atlas_path = self.get_atlas_filepath()
            self.labels_path = self.get_labels_filepath()
        else:
            self.labels_path = self.get_labels_filepath()

    def get_data_filepath(self) -> str:
        filepath = f"{os.getcwd()}/input/dummy_mice_data.csv"
        if self.is_valid_input(filepath):
            return filepath

    def get_coords_filepath(self) -> str:
        filepath = f"{os.getcwd()}/input/mice_coords.csv"
        if self.is_valid_input(filepath):
            return filepath

    def get_atlas_filepath(self) -> str:
        filepath = f"{os.getcwd()}/input/mice_atlas.mnc"
        if self.is_valid_input(filepath):
            return filepath

    def get_labels_filepath(self) -> str:
        filepath = f"{os.getcwd()}/input/mice_labels.csv"
        if self.is_valid_input(filepath):
            return filepath

    @staticmethod
    def is_valid_input(filepath: str):
        if not filepath:
            print("Exiting program.")
            sys.exit(0)
        elif os.path.exists(filepath):
            return True
        else:
            raise TypeError(f"File {filepath} does not exist. Please double check if input path to file is correct.")


class ManualSetup(SetupHelper):
    def __init__(self) -> None:
        self.setup = self.get_parameters()

    def get_parameters(self) -> Setup:
        return Setup(
            alpha=0.05,
            theta=0.95,  # probability map threshold, paper: theta = 1 - alpha
            threshold=0.3,
            weights="pearson_correlation",
            correction_type="fdr_bh",
            criteria_representation="mean",  # "mean", "median", "mode", "geodesic"
            data_balance="imbalanced",
            n_samples_measures=30,
            plot_3d=True,
            brain_type="mice/rat",
            which_plot="heatmap",
            output_format="png",
            mbn_method="ms_scheme",  # "ms_scheme", "conventional"
            n_samples=100,
            random_type="bootstrap",  # "bootstrap", "subsampling"
            interactive=True,
            seed=13,
        )


# TODO write a class to parse setup parameters from a GUI (tkinter, or pyqt)
class TkSetup(SetupHelper):
    def __init__(self) -> None:
        self.setup = self.get_parameters()

    def get_parameters(self) -> Setup:
        pass
