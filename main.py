# GENERAL IMPORT
import json
import warnings
warnings.filterwarnings("ignore")

import argparse
import os, sys

# CUSTOM
from modules import DataAdministrator, FaceAnalyst, FaceRegisterer, DecisionMaker
from modules.utils import *
from captureFrames import captureFrames


def main(config):
  '''
  # Data 
  dataAdministrator = DataAdministrator(args.filepath_image_folder, args.filepath_embedding_data )
  dataLoader, idx_to_class = dataAdministrator.create_data_loader()
  print("\ncompleted DataAdministrator")

  # Face
  faceAnalyst = FaceAnalyst(args.face_prob_threshold1, args.face_prob_threshold2, args.face_dist_threshold, args.focal)
  print("\ncompleted FaceAnalyst")
  data = faceAnalyst.detectFaceFromDataLoader(dataLoader, idx_to_class)
  dataAdministrator.save_embedding_data(data)
  dataAdministrator.load_embedding_data()
  print("\nsaved and loaded embedding data")
  '''

  dataAdministrator = DataAdministrator(config["DataAdministrator"])

  faceAnalyst = FaceAnalyst(config["FaceAnalyst"])
  sys.exit()

  # Registerer
  faceRegisterer = FaceRegisterer()

  # # DecisionMaker
  # decisionMaker = DecisionMaker()

  # # Object Tracking
  # objectTracker = ObjectTracker()
  # print("\ncompleted objectTracking")

  # VideoCapture
  # isregistration = captureFrames(args, dataAdministrator, faceAnalyst, faceRegisterer)
  
  captureFrames(config["default"], dataAdministrator, faceAnalyst)

def parse_args():
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


