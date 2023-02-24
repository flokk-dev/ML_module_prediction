"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose:
"""

# IMPORT: file loading
import pydicom

# IMPORT: tensor
import torch
import torchio as tio


class PostProcessor:
    def __init__(self, params: dict):
        """
        Initializes an instance of PostProcessor class.

        Parameters:
            - params (dict): the inference parameters.
        """
        self._default_spacing = (2., 1.5234375, 1.5234375)
        self._params = params

    def launch(self, volume: torch.Tensor, meta_data: dict) -> torch.Tensor:
        """
        Launches and applies post-processing.

        Parameters:
            - volume (torch.Tensor): the volume to post-process.
            - meta_data (dict): the input file's metadata.

        Returns:
            - (torch.Tensor): the post-processed volume.
        """
        # RESAMPLE
        if meta_data["spacing"] != self._default_spacing:
            volume = self._resample(volume, meta_data["spacing"])

        # CROP TO EXPECTED SHAPE
        volume = self._crop_or_pad(volume, meta_data["shape"])

        # REVERSE IF NOT GOOD POSITION
        if meta_data["position"] != "HFS":
            volume = torch.flip(volume, [0, 1])

        # RESCALE INTENSITY
        if self._params["rescale_intensity"]:
            for slice_idx in range(volume.shape[1]):
                volume[:, slice_idx] = self._rescale_intensity(
                    volume[:, slice_idx], meta_data["files"][slice_idx]
                )

        # CLIP INTENSITY
        if self._params["clip_value"] > 0:
            volume = torch.clip(volume, 0, self._params["clip_value"])

        # CROP OR PAD AND CROP Z-AXIS
        if self._params["crop_value"] > 0:
            volume = self._crop_or_pad(volume, meta_data["shape"], self._params["crop_value"])

        return volume

    def _resample(self, volume: torch.Tensor, target_spacing: tuple) -> torch.Tensor:
        """
        Resamples volume according to the desired spacing.

        Parameters:
            - volume (torch.Tensor): the volume to resample.
            - target_spacing (dict): the desired slice spacing.

        Returns:
            - (torch.Tensor): the resampled volume.
        """
        return tio.Resample(
            target=(
                1 / (self._default_spacing[0] / target_spacing[0]),
                1 / (self._default_spacing[1] / target_spacing[1]),
                1 / (self._default_spacing[2] / target_spacing[2])
            ),
            image_interpolation="linear"
        )(volume)

    @staticmethod
    def _crop_or_pad(volume: torch.Tensor, target_shape: tuple, crop_value: int = 0) -> torch.Tensor:
        """
        Crops or pads volume according to the desired shape.

        Parameters:
            - volume (torch.Tensor): the volume to crop or pad.
            - target_shape (dict): the desired shape.
            - crop_value (int): the number of slices to crop before applying crop or pad.

        Returns:
            - (torch.Tensor): the cropped or padded volume.
        """
        if crop_value > 0:
            volume = volume[:, crop_value:-crop_value]
        return tio.CropOrPad(target_shape, padding_mode=0)(volume).data

    @staticmethod
    def _rescale_intensity(volume: torch.Tensor, dicom_file: pydicom.Dataset):
        """
        Rescales volume's intensity using dicom's metadata.

        Parameters:
            - volume (torch.Tensor): the volume to rescale intensity.
            - dicom_file (pydicom.DicomFile): the dicom file containing metadata.

        Returns:
            - (torch.Tensor): the rescaled volume.
        """
        # Get acquisition and injection time
        rescale_slope = dicom_file.get("RescaleSlope")
        rescale_intercept = dicom_file.get("RescaleIntercept")

        return (volume - rescale_intercept) / rescale_slope
