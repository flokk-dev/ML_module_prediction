"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose:
"""

# IMPORT: utils
import copy
import torch

# IMPORT: data processing
import numpy as np

# IMPORT: project
import utils


class FileLoader:
    def __init__(self):
        """
        Initializes an instance of FileLoader class.
        """
        self._meta_data = dict()

    def get_meta_data(self) -> dict:
        """
        Returns metadata.

        Returns:
            - (dict): the file loader's metadata.
        """
        return copy.deepcopy(self._meta_data)

    def get_shape(self) -> tuple:
        """
        Returns loaded volume shape.

        Returns:
            - (tuple): the loaded volume shape.
        """
        return self._meta_data["shape"]

    def get_spacing(self) -> tuple:
        """
        Returns loaded volume spacing.

        Returns:
            - (tuple): the loaded volume spacing.
        """
        return self._meta_data["spacing"]

    def load(self, file_path: str) -> torch.Tensor:
        """
        Loads file's volume.

        Parameters:
            - file_path (str): the file's path.

        Returns:
            - (dict): the numpy file content as a tensor.
        """
        return torch.unsqueeze(utils.numpy_to_tensor(self._load(file_path)), dim=0)

    def _load(self, file_path: str) -> np.ndarray:
        """
        Loads file's volume.

        Parameters:
            - file_path (str): the file's path.

        Returns:
            - (dict): the numpy file content as a tensor.
        """
        raise NotImplementedError()
