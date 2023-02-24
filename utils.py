"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose:
"""

# IMPORT: utils
import zstd

# IMPORT: data loading
import pydicom

# IMPORT: data processing
import numpy as np
import torch

# IMPORT: data visualization
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def numpy_to_tensor(volume):
    """
    Converts numpy volume into a tensor.

    Parameters:
        - volume (numpy.ndarray): the numpy volume.

    Returns:
        - (torch.Tensor): the volume as a tensor.
    """
    return torch.from_numpy(volume.copy()).type(torch.float32)


def load_numpy_compressed(path: str):
    """
    Loads numpy compressed file.

    Parameters:
        - path (str): the file's path.

    Returns:
        - (torch.Tensor): the numpy file content as a tensor.
    """
    volume = np.load(path, allow_pickle=True)
    header = volume["header"][()]

    volume = zstd.decompress(volume["data"])
    volume = np.frombuffer(volume, dtype=np.float32).copy()
    volume = np.reshape(volume, header["shape"])

    return torch.from_numpy(volume).type(torch.float32)


def load_numpy(path: str):
    """
    Loads numpy file.

    Parameters:
        - path (str): the file's path.

    Returns:
        - (torch.Tensor): the numpy file content as a tensor.
    """
    return torch.from_numpy(np.load(path)).type(torch.float32)


def load_tensor(path: str):
    """
    Loads torch file.

    Parameters:
        - path (str): the file's path.

    Returns:
        - (torch.Tensor): the torch file content as a tensor.
    """
    return torch.load(path).type(torch.float32)


def plot_volume(volume):
    if len(volume.shape) == 4:
        volume = volume[0]

    fig = plt.figure(figsize=(15, 8))

    gs = GridSpec(1, 3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax_tmp = ax0.imshow(torch.max(volume, dim=1).values, cmap="gray")
    ax1.imshow(torch.max(volume, dim=2).values, cmap="gray")
    ax2.imshow(torch.max(volume, dim=0).values, cmap="gray")
    fig.colorbar(ax_tmp)

    plt.tight_layout()
    plt.show()


def get_dicom_field(file_path: str, field: str):
    """
    Returns dicom field.

    Parameters:
        - file_path (str): the dicom file's path.
        - field (str): the dicom field.

    Returns:
        - (): the dicom field's value.
    """
    return pydicom.read_file(file_path).get(field)


def get_dicom_spacing(f_file_path: str, s_file_path: str):
    """
    Returns dicom volume's spacing.

    Parameters:
        - f_file_path (str): the first dicom file's path.
        - s_file_path (str): the second dicom file's path.

    Returns:
        - (): the dicom field's value.
    """
    first = pydicom.read_file(f_file_path)
    second = pydicom.read_file(s_file_path)

    f_spacing = first.get("ImagePositionPatient")
    s_spacing = second.get("ImagePositionPatient")

    if (f_spacing is None) or (s_spacing is None):
        z_spacing = first.get("SliceThickness")
        if z_spacing is None:
            raise ValueError
        else:
            z_spacing = float(z_spacing)
    else:
        z_spacing = abs(round(float(f_spacing[2]) - float(s_spacing[2]), 3))

    x_spacing, y_spacing = first.get("PixelSpacing")
    return float(z_spacing), float(x_spacing), float(y_spacing)
