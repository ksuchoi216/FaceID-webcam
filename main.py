# GENERAL IMPORT
import argparse
import json
import warnings

from captureFrames import captureFrames

# CUSTOM
from modules import FaceAnalyst

# from modules.utils import *

# import os
# import sys


warnings.filterwarnings("ignore")


def main(config):
    """
    Args:
        config (dict): stored basic setting variables.
        please refer to ./config/config files
    """
    # faceAnalyst
    faceAnalyst = FaceAnalyst(config["FaceAnalyst"])

    captureFrames(config["default"], faceAnalyst)


def parse_args():
    """
    Returns:
        dict: parsed arguments
    """
    parser = argparse.ArgumentParser(description="FACE ID")
    parser.add_argument(
        "--new_user_name", help="insert new user name", default="newuser"
    )

    args = parser.parse_args()
    # print_dict(vars(args))
    return args


if __name__ == "__main__":
    path_config = "./configs/config.json"
    with open(path_config) as f:
        config = json.load(f)

    args = parse_args()
    config["DataAdministrator"]["foldername_newuser"] = args.new_user_name

    main(config)
