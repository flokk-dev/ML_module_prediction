"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose:
"""

# IMPORT: data processing
import numpy as np
import torch

# IMPORT: data loading
import SimpleITK as sitk
import pydicom

# IMPORT: project
import utils

from src.loading.file_loader import FileLoader


class DicomLoader(FileLoader):
    def __init__(self):
        """
        Initializes an instance of DicomLoader class.
        """
        super(DicomLoader, self).__init__()
        self._reader = sitk.ImageSeriesReader()

    def get_files(self) -> dict:
        """
        Returns loaded volume's dicom file.

        Returns:
            - (dict): the loaded volume's dicom file.
        """
        return self._meta_data["files"]

    def _load(self, file_path: str) -> np.ndarray:
        """
        Loads dicom directory's volume.

        Parameters:
            - file_path (str): the dicom directory's path.

        Returns:
            - (np.ndarray): the dicom directory's content as a numpy ndarray.
        """
        # Get dicom files path
        files_path = self._get_dicom_files(file_path)

        # Load the input volume and sort by InstanceNumber
        files = self._files_path_as_dict(files_path)
        if not self._is_continuous(files):
            raise ValueError("Il manque des coupes dans le scanner.")

        volume = np.asarray([
            files[i].pixel_array for i in sorted(files.keys())
        ], dtype=np.float32)

        # Store the meta data
        self._meta_data["files"] = files
        self._meta_data["shape"] = volume.shape
        self._meta_data["spacing"] = utils.get_dicom_spacing(files_path[0], files_path[1])
        self._meta_data["position"] = utils.get_dicom_field(files_path[0], "PatientPosition")

        return volume

    def _get_dicom_files(self, path) -> sitk.ImageSeriesReader_GetGDCMSeriesFileNames:
        """
        Returns the dicom files' paths.

        Parameters:
            - path (str): the dicom directory's path.

        Returns:
            - (sitk.ImageSeriesReader_GetGDCMSeriesFileNames): the dicom files' paths.
        """
        dicom_serie = self._reader.GetGDCMSeriesIDs(path)
        if len(dicom_serie) > 1:
            raise ValueError("Too much series in the repertory.")

        return self._reader.GetGDCMSeriesFileNames(path, dicom_serie[0])

    @staticmethod
    def _files_path_as_dict(files_path: sitk.ImageSeriesReader_GetGDCMSeriesFileNames):
        """
        Returns a dictionary with InstanceNumber as key and dicom file as value.

        Parameters:
            - files_path (sitk.ImageSeriesReader_GetGDCMSeriesFileNames): the dicom files' paths.

        Returns:
            - (dict): a dictionary with InstanceNumber as key and dicom file as value.
        """
        tmp_files = dict(map(
            lambda f: (f.get("InstanceNumber") - 1, f),
            map(lambda f: pydicom.read_file(f), files_path)
        ))

        first_idx = min(tmp_files.keys())
        if first_idx == 0:
            return tmp_files

        return {slice_idx-first_idx: file for slice_idx, file in tmp_files.items()}

    @staticmethod
    def _is_continuous(files_dict: dict):
        """
        Verifies if dictionary's keys are continuous or not.

        Parameters:
            - files_dict (dict): a dictionary with InstanceNumber as key and dicom file as value.

        Returns:
            - (dict): True if continuous else False.
        """
        slices_idx = set(files_dict.keys())
        expected_slices_idx = set(range(max(files_dict.keys()) + 1))

        return len(slices_idx.difference(expected_slices_idx)) == 0
