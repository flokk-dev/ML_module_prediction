"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose: Represents Transformer using 2.5D data.
"""

# IMPORT: deep learning
import torch
from monai.networks.nets import SwinUNETR as monai_SwinUNETR


class SwinUNETR(monai_SwinUNETR):
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, weights_path: str = None, in_channels: int = 1):
        """
        Initializes an instance of SwinUNETR class.

        Parameters:
            - weights_path (str): the model weights' path to load.
            - in_channels (int): the number of in channels.
        """
        # Initialise the model
        super(SwinUNETR, self).__init__(img_size=(512, 512), spatial_dims=2,
                                        in_channels=in_channels, out_channels=1,
                                        drop_rate=0.2)

        self.name = "UNet_25D"
        self.weights_path = weights_path

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path))
