"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose: Computes the inference of a given model.
"""

# IMPORT: utils
import os
import time

import zstd
from tqdm import tqdm

# IMPORT: deep learning
import torch

# IMPORT: projet
import paths

from .loading import DicomLoader, NRRDLoader
from .image_processing import PreProcessor, PostProcessor
from .patching import Patcher
from .models import SwinUNETR


class PredictionManagement:
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, file_path: str, file_type: str, params: dict):
        """
        Initializes an instance of PredictionManagement class.

        Parameters:
            - file_path (str): the input file path.
            - file_type (str): the input file type.
            - params (dict): the inference parameters.
        """
        # Parameters
        self._file_path = file_path
        self._params = params

        self._patch_height = 5
        self._batch_size = 16

        self._model = SwinUNETR(
            weights_path=paths.MODEL_PATH,
            in_channels=self._patch_height
        ).to(torch.device(self._DEVICE))

        # Loader
        self._loader = DicomLoader() if file_type == "dicom" else NRRDLoader()

        # Pre-processor
        self._pre_processor = PreProcessor()

        # Patcher
        self._patcher = Patcher(patch_height=self._patch_height)

        # Post-processor
        self._post_processor = PostProcessor(self._params)

    def launch(self):
        """
        Launches inference process.
        """
        # Clear GPU cache
        torch.cuda.empty_cache()

        # Loading
        print("\nChargement des fichiers")

        start = time.time()
        input_volume = self._loader.load(self._file_path)
        print(f"{round(time.time() - start, 3)} secondes.")

        # Pre-processing
        print("\nPre-processing de l'examen TEP")

        start = time.time()
        input_volume = self._pre_processor.launch(input_volume, self._loader.get_meta_data())
        print(f"{round(time.time() - start, 3)} secondes.")

        # Patching
        print("\nFormatage des données en 2.5D")

        start = time.time()
        input_patches = self._patcher.generate_patches(input_volume)
        print(f"{round(time.time() - start, 3)} secondes.")

        # Segmentation
        print("\nEstimation du bruit")

        start = time.time()
        prediction = self._predict_noise(input_patches)
        print(f"{round(time.time() - start, 3)} secondes.")

        # Post-processing
        print("\nPost-processing de l'estimation")

        start = time.time()
        prediction = self._post_processor.launch(prediction, self._loader.get_meta_data())
        print(f"{round(time.time() - start, 3)} secondes.")

        # Saving
        print("\nSauvegarde de l'estimation")
        self._save(prediction)

    def _predict_noise(self, input_volume: torch.Tensor) -> torch.Tensor:
        """
        Predicts volume's noise.

        Parameters:
            - input_volume (torch.Tensor): the volume to predict noise from.

        Returns:
            - (torch.Tensor): the predicted noise as a tensor.
        """
        prediction = torch.Tensor()
        for i in tqdm(range(0, input_volume.shape[0], self._batch_size)):
            batch = input_volume[i: i + self._batch_size].to(torch.device(self._DEVICE))

            probs = self._model(batch).detach()
            prediction = torch.cat((prediction, probs.cpu()), dim=0)

        prediction = torch.movedim(prediction, 0, 1)
        return self._patcher.aggregate_patches(prediction).cpu()

    def _save(self, volume: torch.Tensor):
        """
        Saves volume's noise.

        Parameters:
            - volume (torch.Tensor): the volume to save.
        """
        torch.save(volume, os.path.join(os.path.dirname(self._file_path), f"noise_volume.pt"))
