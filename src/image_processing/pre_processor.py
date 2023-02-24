"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose:
"""

# IMPORT: tensor
import torch
import torchio as tio


class PreProcessor:
    def __init__(self):
        """
        Initializes an instance of PreProcessor class.
        """
        self._default_spacing = (2., 1.5234375, 1.5234375)
        self._desired_shape = 512

    def launch(self, volume: torch.Tensor, meta_data: dict):
        """
        Launches and applies pre-processing.

        Parameters:
            - volume (torch.Tensor): the volume to pre-process.
            - meta_data (dict): the input file's metadata.

        Returns:
            - (torch.Tensor): the pre-processed volume.
        """
        # RESCALE INTENSITY
        for slice_idx in range(volume.shape[1]):
            volume[:, slice_idx] = self._rescale_intensity(
                volume[:, slice_idx], meta_data["files"][slice_idx]
            )

        # REVERSE IF NOT GOOD POSITION
        if meta_data["position"] != "HFS":
            volume = torch.flip(volume, [0, 1])

        # RESAMPLE
        if meta_data["spacing"] != self._default_spacing:
            volume = self._resample(volume, meta_data["spacing"])

        # CROP OR PAD
        if meta_data["shape"] != (meta_data["shape"][0], self._desired_shape, self._desired_shape):
            volume = self._crop_or_pad(volume)

        return volume

    def _resample(self, volume, input_spacing):
        """
        Resamples volume according to the desired spacing.

        Parameters:
            - volume (torch.Tensor): the volume to resample.
            - input_spacing (dict): the desired slice spacing.

        Returns:
            - (torch.Tensor): the resampled volume.
        """
        return tio.Resample(
            target=(
                1 / (input_spacing[0] / self._default_spacing[0]),
                1 / (input_spacing[1] / self._default_spacing[1]),
                1 / (input_spacing[2] / self._default_spacing[2])
            ),
            image_interpolation="linear"
        )(volume)

    def _crop_or_pad(self, volume):
        """
        Crops or pads volume according to the desired shape.

        Parameters:
            - volume (torch.Tensor): the volume to crop or pad.

        Returns:
            - (torch.Tensor): the cropped or padded volume.
        """
        return tio.CropOrPad(
            (volume.shape[1], self._desired_shape, self._desired_shape), padding_mode=0
        )(volume).data

    @staticmethod
    def _rescale_intensity(volume, dicom_file):
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

        return (volume * rescale_slope) + rescale_intercept
