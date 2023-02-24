"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose:
"""

# IMPORT: data loading
import nrrd

# IMPORT: data processing
import numpy as np

# IMPORT: project
from src.loading.file_loader import FileLoader


class NRRDLoader(FileLoader):
    def __init__(self):
        """
        Initializes an instance of NRRDLoader class.
        """
        super(NRRDLoader, self).__init__()

    def _load(self, file_path):
        """
        Loads file's volume.

        Parameters:
            - file_path (str): the file's path.

        Returns:
            - (np.ndarray): the file content as a numpy ndarray.
        """
        # Load the input volume
        volume, header = nrrd.read(file_path)

        # Store the meta data
        shape = header["sizes"]
        self._meta_data["shape"] = (shape[0], shape[1], shape[2])

        spacing = header["space directions"]
        self._meta_data["spacing"] = (spacing[2][2], spacing[0][0], spacing[1][1])

        return self._adjust_axis(volume)

    @staticmethod
    def _adjust_axis(volume):
        """
        Adjusts volume's axis.

        Parameters:
            - volume (np.ndarray): the volume to adjust axis.

        Returns:
            - (np.ndarray): the adjusted volume.
        """
        return np.rot90(volume, k=1, axes=(0, 2)).copy()
