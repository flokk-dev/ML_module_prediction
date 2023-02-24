"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose:
"""

# IMPORT: tensor
import torch

# IMPORT: project
import utils


class Patcher:
    def __init__(self, patch_height: int = 1):
        """
        Initializes an instance of Patcher class.

        Parameters:
            - patch_height (int): the patch's height.
        """
        self._patch_height = patch_height

    def generate_patches(self, volume):
        """
        Generates patches from a volume.

        Parameters:
            - volume (torch.Tensor): the volume to generate patches from.

        Returns:
            - (torch.Tensor): the generated patches.
        """
        input_patches = torch.squeeze(volume)
        input_patches = input_patches.unfold(0, self._patch_height, 1)
        return torch.movedim(input_patches, 3, 1)

    def aggregate_patches(self, patches):
        """
        Aggregates patches into a volume.

        Parameters:
            - patches (torch.Tensor): the patches to aggregate.

        Returns:
            - (torch.Tensor): the aggregated volume.
        """
        shape = list(patches.shape)
        shape[1] = self._patch_height // 2

        return torch.cat((torch.zeros(shape), patches, torch.zeros(shape)), dim=1)
