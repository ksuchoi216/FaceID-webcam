
# from sqlalchemy import true
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import torch
import cv2
from PIL import Image

class DataAdministrator():  
  def __init__(self, config):  
    self.filepath_image_folder = config["filepath_image_folder"]
    self.filepath_embedding_data = config["filepath_embedding_data"]
    self.filepath_newuser_image = config["filepath_newuser_image"]
    self.foldername_newuser = config["foldername_newuser"]
    self.the_number_of_frames = config["the_number_of_frames"]
    self.interval_between_frames = config["interval_between_frames"]
    self.frame_counter = 0
    self.interval_counter = 0 

  def load_embedding_data(self):
    try:
      self.embedding_data = torch.load(self.filepath_embedding_data)
    except:
      print("There is no saved data file. Please check the data file")


  def create_data_loader(self):
    # image load
    dataset = datasets.ImageFolder(self.filepath_image_folder)
    idx_to_class = {i:c for c,i in dataset.class_to_idx.items()}

    dataLoader = DataLoader(dataset, collate_fn = lambda x: x[0])

    return dataLoader, idx_to_class
  
  def save_embedding_data(self, data):
    torch.save(data, self.filepath_embedding_data)
    print("saved embedding data in ", self.filepath_embedding_data)

  def get_embedding_data(self):
    return self.embedding_data

  def createNewDirectory(self):
    new_path = os.path.join(self.filepath_newuser_image, self.foldername_newuser)
    if not os.path.exists(new_path):
      os.makedirs(new_path)
    else:
      print("The directory is already created.")

  def saveFrameInFolder(self, org_frame):
    if self.frame_counter == self.the_number_of_frames:
      self.frame_counter = 0
      return False

    elif self.interval_counter == 0:
      self.frame_counter += 1
      file_name = str(self.frame_counter) + '.png'
      file_path = self.filepath_newuser_image + self.foldername_newuser+ '/' + file_name
      cv2.imwrite(file_path, org_frame)

      print('the image was successfully saved in '+file_path)
      return True

    else:
      self.interval_counter += 1
      return True


  

  
