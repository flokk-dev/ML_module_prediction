"""
Creator: HOCQUET Florian
Date: 11/01/2023
Version: 1.1

Purpose: Manages the project.
"""

# IMPORT: utils
import time
import argparse

# IMPORT: projet
from src import PredictionManagement

# WARNINGS SHUT DOWN
import warnings
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    """
    Parses shell's arguments.

    Returns:
        - (argparse.Namespace): the object containing all arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", type=str, nargs="?",
                        default="None", help="file path.")

    parser.add_argument("-t", "--file_type", type=str, nargs="?",
                        choices=["dicom", "nrrd"], default="dicom", help="file type.")

    parser.add_argument("-ri", "--rescale_intensity", type=bool, nargs="?",
                        default=False, help="rescale intensity to its input range.")

    parser.add_argument("-cl", "--clip_value", type=int, nargs="?",
                        default=0, help="clip volume intensity according to clip value (max).")

    parser.add_argument("-cr", "--crop_value", type=int, nargs="?",
                        default=0, help="crop z-axis volume according to crop value.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.file == "None" or len([arg for arg in args._get_kwargs() if arg[1]]) > 4:
        raise ValueError(f"Wrong arguments, use -help to have more information.")

    params = {"rescale_intensity": args.rescale_intensity, "clip_value": args.clip_value, "crop_value": args.crop_value}
    predictor = PredictionManagement(file_path=args.file, file_type=args.file_type, params=params)

    start = time.time()
    predictor.launch()
    print(f"Temps total de segmentation: {round(time.time() - start, 3)} secondes.")
