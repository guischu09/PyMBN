import itertools
import os
import shutil
import sys
from typing import Union

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import open3d as o3d
from mne_connectivity.viz import plot_connectivity_circle
from skimage import measure

from .base_classes import Setup
from .data_importer import PetData
import logging


def get_output_dir(clobber: bool = False) -> str:
    output_dir = os.path.join(os.getcwd(), "outputs")
    if clobber and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_results_dir(clobber: bool = False) -> str:
    results_dir = os.path.join(os.getcwd(), "results")
    if clobber and os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


RESULTS_DIR = get_results_dir(clobber=False)
OUTPUT_DIR = get_output_dir(clobber=False)
NODE_SIZE_CTE = 1


class NetworkPlotter:
    def __init__(
        self,
        networks: np.ndarray,
        data: PetData,
        labels_path: str,
        atlas_path: str,
        coords_path: str,
        setup: Setup,
        clobber: bool = True,
    ) -> None:
        self.setup = setup
        self.networks = networks
        self.labels_path = labels_path
        self.atlas_path = atlas_path
        self.coords_path = coords_path
        self._set_output_dir(clobber)
        self._set_results_dir(clobber)
        self._set_group_names(data)

    def _set_group_names(self, data):
        group_names = []
        for d in data:
            group_names.append(d[1])
        self.group_names = group_names

    def _set_output_dir(self, clobber):
        output_dir = os.path.join(os.getcwd(), "outputs")
        if clobber and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        self.out_dir = output_dir

    def _set_results_dir(self, clobber):
        results_dir = os.path.join(os.getcwd(), "results")
        if clobber and os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        self.results_dir = results_dir

    def plot_networks_heatmaps(self, cmap: str = "turbo") -> None:
        self._set_vmin()

        for g, group in enumerate(self.group_names):
            network = self.networks[:, :, g]
            output_path = os.path.join(self.results_dir, f"{group}_heatmap.{self.setup.output_format}")
            logging.info(f">> Plotting heatmap for group {group}")
            plot_heatmap(network=network, labels=self.labels_path, output_path=output_path, v_min=self.v_min)
            logging.info(f">> Heatmap saved to {output_path}")

    def plot_networks_2d(
        self, facecolor: str = "white", textcolor: str = "black", cmap: str = "turbo", interactive: bool = False
    ) -> None:

        for g, group in enumerate(self.group_names):
            network = self.networks[:, :, g]
            network = network - np.eye(network.shape[0])
            output_path = os.path.join(self.results_dir, f"{group}_circle_plot.{self.setup.output_format}")
            labels = list(pd.read_csv(self.labels_path, header=None)[0])
            fig, _ = plot_connectivity_circle(
                con=network,
                node_names=labels,
                colormap=cmap,
                vmin=self.v_min,
                vmax=1,
                colorbar=True,
                facecolor=facecolor,
                textcolor=textcolor,
                show=interactive,
            )
            fig.savefig(output_path, facecolor=facecolor)
            plt.close()

    def plot_networks_brain3d(self, cmap: str = "turbo", interactive: bool = False):

        for g, group in enumerate(self.group_names):
            network = self.networks[:, :, g]
            output_path = os.path.join(self.results_dir, f"{group}_brain.{self.setup.output_format}")

            plot_3d(
                atlas=self.atlas_path,
                labels=self.labels_path,
                coordinates=self.coords_path,
                network=network,
                brain_type=self.setup.brain_type,
                output_path=output_path,
                v_min=self.v_min,
                cmap=cmap,
                interactive=interactive,
            )
        if interactive:
            sys.exit(0)

    def _set_vmin(self):
        unique_set = set(self.networks.ravel())
        unique_set.remove(0)
        self.v_min = min(unique_set)


def get_vmin(networks: np.ndarray) -> float:
    unique_set = set(networks.ravel())
    unique_set.remove(0)
    return min(unique_set)


def get_coords_dict(coords_path: str) -> dict:
    coords = pd.read_csv(coords_path, header=None)
    # Creates a dict where the keys are rois and values are x, y and z coordinates
    coordinates = {}
    for k in range(coords.shape[0]):
        coordinates[coords.iloc[k, 0]] = np.array([coords.iloc[k, 1], coords.iloc[k, 2], coords.iloc[k, 3]])
    return coordinates


def get_group_names(data: PetData) -> list[str]:
    group_names = []
    for d in data:
        group_names.append(d[1])
    return group_names


def plot_networks_3d(
    data: PetData,
    networks: np.ndarray,
    labels_path: str,
    atlas_path: str,
    coords_path: str,
    brain_type: str,
    output_format: str = "png",
    interactive: bool = False,
    min_value: float = None,
    color_map: str = "turbo",
    results_dir: str = RESULTS_DIR,
) -> None:

    output_format = output_format.lstrip(".").lower()

    if min_value is None:
        min_value = get_vmin(networks)

    if output_format != "png":
        logging.warning("3D plots are only available in png format.")

    group_names = get_group_names(data)

    for g, group in enumerate(group_names):
        network = networks[:, :, g]
        output_path = os.path.join(results_dir, f"{group}_brain.{output_format}")

        plot_3d(
            atlas=atlas_path,
            labels=labels_path,
            coordinates=coords_path,
            network=network,
            brain_type=brain_type,
            output_path=output_path,
            v_min=min_value,
            cmap=color_map,
            interactive=interactive,
        )
    if interactive:
        sys.exit(0)


def plot_networks_2d(
    data: PetData,
    networks: np.ndarray,
    labels_path: str,
    output_format: str = "png",  # png, pdf, svg
    interactive: bool = False,
    min_value: float = None,
    color_map: str = "turbo",
    results_dir: str = RESULTS_DIR,
    facecolor: str = "white",
    textcolor: str = "black",
) -> None:

    output_format = output_format.lstrip(".").lower()

    if min_value is None:
        min_value = get_vmin(networks)

    group_names = get_group_names(data)

    for g, group in enumerate(group_names):
        network = networks[:, :, g]
        network = network - np.eye(network.shape[0])
        output_path = os.path.join(results_dir, f"{group}_circle_plot.{output_format}")
        labels = list(pd.read_csv(labels_path, header=None)[0])
        fig, _ = plot_connectivity_circle(
            con=network,
            node_names=labels,
            colormap=color_map,
            vmin=min_value,
            vmax=1,
            colorbar=True,
            facecolor=facecolor,
            textcolor=textcolor,
            show=interactive,
        )
        fig.savefig(output_path, facecolor=facecolor)
        plt.close()


def plot_networks_heatmaps(
    data: PetData,
    networks: np.ndarray,
    labels_path: str,
    output_format: str = "png",  # png, pdf, svg
    color_map: str = "turbo",
    min_value: float = None,
    results_dir: str = RESULTS_DIR,
) -> None:

    output_format = output_format.lstrip(".").lower()

    if min_value is None:
        min_value = get_vmin(networks)

    group_names = get_group_names(data)

    for g, group in enumerate(group_names):
        network = networks[:, :, g]
        output_path = os.path.join(results_dir, f"{group}_heatmap.{output_format}")
        plot_heatmap(network=network, labels=labels_path, output_path=output_path, v_min=min_value, cmap=color_map)
        logging.info(f">> Heatmap image saved to {output_path}")


def plot_heatmap(
    network: Union[np.ndarray, str],
    labels: Union[list, str],
    output_path: str,
    v_min: float = 0,
    cmap: str = "turbo",
) -> None:
    if type(labels) == str:
        labels = list(pd.read_csv(labels, header=None)[0])
    elif type(labels) != list:
        raise ValueError("Non supported type. Supported types are List[str] and str.")

    if type(network) == np.ndarray:
        network = pd.DataFrame(network, columns=labels, index=labels)
    elif type(network) == str:
        network = pd.read_csv(network).values
    else:
        raise ValueError("Non supported type. Supported types are np.ndarray and str.")

    mask = np.isin(network, 0)

    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot heatmap figure. We believe that setting the labels in the x axis looks prettier.
    # Change xticklabels to True if you want to display the labels in the x axis of the heatmap as well.
    sns.heatmap(network, mask=mask, vmin=v_min, vmax=1, cmap=cmap, linecolor="k", linewidths=0.05, xticklabels=False)

    ax.set_ylabel("")
    ax.set_xlabel("")

    plt.yticks(rotation=0)
    plt.xticks(rotation=45)

    # Saves figure
    fig.savefig(output_path)

    plt.close()


def plot_2d(
    network_path: str,
    labels_path: str,
    output_path: str = None,
    facecolor: str = "white",
    textcolor: str = "black",
    cmap: str = "turbo",
    interactive: bool = None,
) -> None:
    if interactive is None:
        interactive = False

    network = pd.read_csv(network_path).iloc[:, 1::].values
    network = network - np.eye(network.shape[0])

    unique_set = set(network.ravel())
    unique_set.remove(0)
    v_min = min(unique_set)

    labels = list(pd.read_csv(labels_path, header=None)[0])

    if output_path is None:
        output_path = network_path.replace(".csv", "_circle.png")
    # norm = mpl.colors.Normalize(vmin=v_min, vmax=1)
    # m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, _ = plot_connectivity_circle(
        con=network,
        node_names=labels,
        colormap=cmap,
        vmin=v_min,
        vmax=1,
        colorbar=True,
        facecolor=facecolor,
        textcolor=textcolor,
        show=interactive,
    )
    fig.savefig(output_path, facecolor=facecolor)
    plt.close()


def plot_3d(
    atlas: Union[nib.minc1.Minc1Image, str],
    labels: Union[list, str],
    coordinates: Union[dict, str],
    network: Union[np.ndarray, str],
    brain_type: str,
    output_path: str = "brain.png",
    v_min: float = 0,
    cmap: str = "turbo",
    interactive: bool = False,
) -> None:
    if type(atlas) == str:
        atlas = nib.load(atlas)
    elif type(atlas) != nib.minc1.Minc1Image:
        raise ValueError("Non supported type. Supported types are nib.minc1.Minc1Image and str.")

    if type(labels) == str:
        labels = list(pd.read_csv(labels, header=None)[0])
    elif type(labels) != list:
        raise ValueError("Non supported type. Supported types are List[str] and str.")

    if type(coordinates) == str:
        coordinates = get_coords_dict(coordinates)
    elif type(coordinates) != dict:
        raise ValueError("Non supported type. Supported types are Dict and str.")

    if type(network) == np.ndarray:
        network = pd.DataFrame(network, columns=labels, index=labels)
    elif type(network) == str:
        network = pd.read_csv(network).values
    else:
        raise ValueError("Non supported type. Supported types are np.ndarray and str.")

    norm = mpl.colors.Normalize(vmin=v_min, vmax=1)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Obtain the non repeated pairs between rois
    pairs = np.array(list(itertools.combinations(labels, 2)))

    # Get Voxel Size:
    x_step = atlas.affine[0, 2]
    y_step = atlas.affine[1, 1]
    z_step = atlas.affine[2, 0]

    # Get Origin of image space
    x_start = atlas.affine[0, 3]
    y_start = atlas.affine[1, 3]
    z_start = atlas.affine[2, 3]

    # Get number os slices in the x plane
    x_dim = atlas.shape[2]

    # Get numpy array data and interchange axes to the X, Y, Z order
    brain = atlas.get_fdata()
    brain = np.swapaxes(brain, 0, 2)

    # discretize volume surface
    verts, faces, _, _ = measure.marching_cubes(
        brain, 10, spacing=(abs(x_step), abs(y_step), abs(z_step)), method="lorensen"
    )

    # Converts verts to three axis
    x, y, z = zip(*verts)

    # Adjusts coordinates to our atlas
    for k in coordinates.keys():
        coordinates[k][0] = coordinates[k][0] - (abs(x_start) + (abs(x_step) * -1) * x_dim)
        coordinates[k][1] = coordinates[k][1] - y_start
        coordinates[k][2] = coordinates[k][2] - z_start

    # Gets x, y and z coordinates of all rois
    Xn = [coordinates[k][0] for k in coordinates.keys()]  # x-coordinates of nodes
    Yn = [coordinates[k][1] for k in coordinates.keys()]  # y-coordinates
    Zn = [coordinates[k][2] for k in coordinates.keys()]  # z-coordinates

    # Size of nodes are associated with the node degree
    sizes = []
    strength = {}
    degree = {}
    for i, voi in enumerate(labels):
        sizes.append(np.sum(network.iloc[i] != 0) * 1.0)
        strength[voi] = np.sum(network.iloc[i])
        degree[voi] = np.sum(network.iloc[i] != 0) - 1

    adjSizes = []
    # Adjust sizes
    for i in range(len(labels)):
        adjSizes.append(sizes[i] ** 1.7 + NODE_SIZE_CTE)

    if brain_type == "mice/rat":
        NODE_MAX_SIZE = 0.65
    if brain_type == "human":
        NODE_MAX_SIZE = 8

    adjSizes = adjSizes / np.max(adjSizes) * NODE_MAX_SIZE

    ##### Open3D scene ###
    if brain_type == "mice/rat":
        view = (-55, 54, 71)  # mayavi-style (azimuth, elevation, distance) that looks well for rodent brains
        rtube = 0.03
        node_color = (1.0, 1.0, 1.0)
    elif brain_type == "human":
        view = (180, 90, 546)
        rtube = 0.08
        node_color = (0.0, 0.0, 0.58)
    else:
        raise ValueError("Non supported brain_type. Supported types are 'mice/rat' and 'human'.")

    # Brain surface mesh
    brain_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts, dtype=float)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    brain_mesh.compute_vertex_normals()
    brain_mesh.paint_uniform_color((0.82, 0.82, 0.82))

    # Edges as tubes, colored by correlation strength
    edges_mesh = o3d.geometry.TriangleMesh()
    for p in pairs:
        correlation = network[p[0]][p[1]]
        if correlation != 0:
            color_rgb = m.to_rgba(correlation)[0:3]
            edges_mesh += _make_tube(coordinates[p[0]], coordinates[p[1]], rtube, color_rgb)

    # Nodes as spheres, sized by degree
    nodes_mesh = o3d.geometry.TriangleMesh()
    for i in range(len(labels)):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=adjSizes[i] / 2, resolution=20)
        sphere.translate((Xn[i], Yn[i], Zn[i]))
        sphere.paint_uniform_color(node_color)
        nodes_mesh += sphere
    edges_mesh.compute_vertex_normals()
    nodes_mesh.compute_vertex_normals()

    # Camera: look at the brain center from the requested direction, at a distance that fits the brain
    bbox = brain_mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    radius = np.linalg.norm(bbox.get_extent()) / 2
    distance = 1.1 * radius / np.sin(np.radians(CAMERA_FOV_DEG / 2))
    eye = _view_to_eye(view[0], view[1], distance, focal=center)

    if interactive:
        o3d.visualization.draw_geometries(
            [brain_mesh, edges_mesh, nodes_mesh], window_name=os.path.basename(output_path), width=1000, height=1000
        )
    else:
        _render_offscreen(brain_mesh, edges_mesh, nodes_mesh, center, eye, output_path)
        logging.info(f">> 3D brain image saved to {output_path}")


CAMERA_FOV_DEG = 30.0


def _view_to_eye(azimuth: float, elevation: float, distance: float, focal: np.ndarray) -> np.ndarray:
    """Convert a mayavi-style view (azimuth about z, elevation from z, distance) into a camera position."""
    az, el = np.radians(azimuth), np.radians(elevation)
    direction = np.array([np.cos(az) * np.sin(el), np.sin(az) * np.sin(el), np.cos(el)])
    return np.asarray(focal) + distance * direction


def _make_tube(p0, p1, radius: float, color) -> o3d.geometry.TriangleMesh:
    """Cylinder from p0 to p1 with the given radius and RGB color."""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = np.linalg.norm(direction)
    tube = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=12, split=1)
    if length > 0:
        z = np.array([0.0, 0.0, 1.0])
        d = direction / length
        axis = np.cross(z, d)
        s = np.linalg.norm(axis)
        angle = np.arctan2(s, np.dot(z, d))
        if s > 1e-8:
            tube.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * angle), center=(0, 0, 0))
        elif angle > np.pi / 2:  # anti-parallel
            tube.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0))
    tube.translate((p0 + p1) / 2)
    tube.paint_uniform_color(color)
    return tube


def _render_offscreen(
    brain_mesh, edges_mesh, nodes_mesh, center, eye, output_path: str, size: int = 2000, lit: bool = True
) -> None:
    """Render the scene headlessly (EGL) and save it as an image."""
    from open3d.visualization import rendering

    renderer = rendering.OffscreenRenderer(size, size)
    scene = renderer.scene
    scene.set_background([1.0, 1.0, 1.0, 1.0])
    scene.view.set_post_processing(False)
    scene.scene.set_sun_light([0.3, 0.2, -1.0], [1.0, 1.0, 1.0], 120000)
    scene.scene.enable_sun_light(True)
    scene.scene.set_indirect_light_intensity(60000)

    brain_mat = rendering.MaterialRecord()
    brain_mat.shader = "defaultLitTransparency"
    brain_mat.base_color = [0.82, 0.82, 0.82, 0.12]

    solid_mat = rendering.MaterialRecord()
    solid_mat.shader = "defaultLit" if lit else "defaultUnlit"

    if len(edges_mesh.vertices) > 0:
        scene.add_geometry("edges", edges_mesh, solid_mat)
    scene.add_geometry("nodes", nodes_mesh, solid_mat)
    scene.add_geometry("brain", brain_mesh, brain_mat)

    renderer.setup_camera(CAMERA_FOV_DEG, np.asarray(center), np.asarray(eye), [0.0, 0.0, 1.0])
    image = renderer.render_to_image()
    o3d.io.write_image(output_path, image)
