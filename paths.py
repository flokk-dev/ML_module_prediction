"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose: Manages the project's constants paths.
"""

# IMPORT: utils
import os

"""
ROOT
"""
ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)))

"""
RESOURCES
"""
RESOURCES_PATH = os.path.join(ROOT_PATH, "resources")

"""
MODELS
"""
MODELS_PATH = os.path.join(RESOURCES_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "model.pt")
