import os
import json

import cv2
import torch
import pandas as pd

from external_library.sort.sort import Sort


class ObjectTracker():
  def __init__(self):
    self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    self.object_detector.float()
    self.object_detector.eval()
    
    self.mot_tracker = Sort()
    
  def track_objects(self, image):
    results = self.object_detector(image)    
    df = results.pandas().xyxy[0]
    detections = df[df['name']=='person'].drop(columns='name').to_numpy()
    track_ids = self.mot_tracker.update(detections)
    
    for i in range(len(track_ids.tolist())):
      coords = track_ids.tolist()[i]
      xmin, ymin, xmax, ymax = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
      name_idx = int(coords[4])
      name = 'ID: {}'.format(str(name_idx))

      image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0 ,0), 2)
      image = cv2.putText(image, name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
      