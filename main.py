# GENERAL IMPORT
import json
import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys

# CUSTOM
from modules import FaceAnalyst
from modules.utils import *
from captureFrames import captureFrames


def main(config):
  """_summary_

  Args:
      config (dict): stored basic setting variables. please refer to ./config/config files
  """
  # dataAdmin
  # dataAdministrator = DataAdministrator(config["DataAdministrator"])
  # dataLoader, idx_to_class = dataAdministrator.create_data_loader()
  # print(dataLoader, idx_to_class)

  # faceAnalyst
  faceAnalyst = FaceAnalyst(config["FaceAnalyst"])
  # data = faceAnalyst.detectFaceFromDataLoader(dataLoader, idx_to_class)
  
  # loading embedding data
  # dataAdministrator.save_embedding_data(data)
  
  captureFrames(config["default"], faceAnalyst)

def parse_args():
  """_summary_

  Returns:
      dict: parsed arguments
  """
  parser = argparse.ArgumentParser(description = "FACE ID")
  parser.add_argument("--new_user_name", help="insert new user name", default = "newuser")

  args = parser.parse_args()
  print_dict(vars(args))
  return args

if __name__ == "__main__":
  path_config = "./configs/config.json"
  with open(path_config) as f:
    config = json.load(f)

  args = parse_args()
  config["DataAdministrator"]["foldername_newuser"] = args.new_user_name

  main(config)


