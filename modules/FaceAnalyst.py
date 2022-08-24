
import os
import sys
from tabnanny import check
from tkinter.tix import Y_REGION
import torch
import torch.nn as nn
from torch.nn import functional as F
from .external_library import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

# head pose estimation
from .external_library.HeadPoseEstimation.drawFace import draw
from .external_library.HeadPoseEstimation import reference_world as world
import dlib
import mediapipe as mp

from .utils import printd, draw_center_border

def calculateDistance(A, B):
  return torch.dist(A, B).item()

def calculateSimilarity(A, B):
  cos = torch.nn.CosineSimilarity(dim=1)
  return cos(A, B).item()

class FaceAnalyst():
  def __init__(self, cfg):
    customed_pretrained_model = cfg["model"]["customed_pretrained_model"]
    image_size_for_face_detector = cfg["model"]["image_size_for_face_detector"]
    path_for_pretrained_model = cfg["model"]["path_for_pretrained_model"]
    
    self.single_face_detector = MTCNN(image_size=image_size_for_face_detector, margin=0, keep_all=False, min_face_size=40) # keep_all=False
    self.multi_faces_detector = MTCNN(image_size=image_size_for_face_detector, margin=0, keep_all=True, min_face_size=40) # keep_all=True
    self.face_feature_extractor = InceptionResnetV1(pretrained='vggface2')
    # self.object_detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    if customed_pretrained_model is True:
      self.face_feature_extractor.load_state_dict(torch.load(path_for_pretrained_model))
      self.face_feature_extractor.classify = True
      
    self.face_feature_extractor.eval()

    self.face_prob_threshold1 = cfg["face_prob_threshold1"]
    self.face_prob_threshold2 = cfg["face_prob_threshold2"]
    self.face_dist_threshold = cfg["face_dist_threshold"]

    self.transform = transforms.Compose([transforms.Resize((224,224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225],
                                          ),]) 

    # head pose estimation
    self.focal = cfg["focal"]
    self.mp_face_detection = mp.solutions.face_detection
    self.mp_drawing = mp.solutions.drawing_utils
    self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    PREDICTOR_PATH = os.path.join("./modules/external_library/HeadPoseEstimation/models/", "shape_predictor_68_face_landmarks.dat")
    if not os.path.isfile(PREDICTOR_PATH):
      print("PREDICTOR_PATH: ", PREDICTOR_PATH)
      print("[ERROR] USE models/downloader.sh to download the predictor")
      sys.exit()
    self.predictor = dlib.shape_predictor(PREDICTOR_PATH)
    
    self.face3Dmodel = world.ref3DModel()

    self.center_area_size_half = cfg["center_area_size_half"]


  def estimateHeadPose(self, org_image, image, absx, absy, abswidth, absheight):
    img = org_image
    newrect = dlib.rectangle(absx,absy,abswidth,absheight)
    cv2.rectangle(image, (absx, absy), (abswidth, absheight),(0, 255, 0), 2)
    shape = self.predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), newrect)

    draw(image, shape)

    refImgPts = world.ref2dImagePoints(shape)

    height, width, channels = img.shape
    focalLength = self.focal * width
    cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

    mdists = np.zeros((4, 1), dtype=np.float64)

    # calculate rotation and translation vector using solvePnP
    success, rotationVector, translationVector = cv2.solvePnP(
        self.face3Dmodel, refImgPts, cameraMatrix, mdists)

    noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
    noseEndPoint2D, jacobian = cv2.projectPoints(
        noseEndPoints3D, rotationVector, translationVector, cameraMatrix, mdists)

    #  draw nose line
    p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
    p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
    cv2.line(image, p1, p2, (110, 220, 0), thickness=2, lineType=cv2.LINE_AA)

    # calculating euler angles
    rmat, jac = cv2.Rodrigues(rotationVector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    # print('*' * 80)
    # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")
    x = np.arctan2(Qx[2][1], Qx[2][2])
    y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
    z = np.arctan2(Qz[0][0], Qz[1][0])
    # print("ThetaX: ", x)
    # print("ThetaY: ", y)
    # print("ThetaZ: ", z)
    # print('*' * 80)

    if angles[1] < -15:
        GAZE = "Looking: Left"
    elif angles[1] > 15:
        GAZE = "Looking: Right"
    else:
        GAZE = "Forward"

    cv2.putText(image, str(round(angles[1],4)), (absx, absy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
    cv2.putText(image, GAZE, (absx, absheight+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)

    return image


  def detectFaceFromDataLoader(self, dataLoader, idx_to_class):
    name_list = []
    embedding_list = []

    for img, idx in dataLoader:
      face, prob = self.single_face_detector(img, return_prob = True)

      if face is not None and prob > self.face_prob_threshold1:
        emb = self.face_feature_extractor(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])
      
    data = [embedding_list, name_list]

    return data

  def detectFacesFromFrame(self, frame):
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = self.multi_faces_detector(img, return_prob=True)

    return img, img_cropped_list, prob_list
  
  def identifyFace(self, embedding_data, org_image, image, absx, absy, abswidth, absheight):
    embedding_list = embedding_data[0] 
    name_list = embedding_data[1]

    croped_face_image = org_image[absy : absheight, absx: abswidth]
    croped_face_image = Image.fromarray(croped_face_image)
    croped_face_image = self.transform(croped_face_image)
    
    face, prob = self.single_face_detector(croped_face_image)

    if face is not None and prob > 0.92:
      emb =  self.face_feature_extractor(face.unsqueeze(0))
      print(f'emb: {emb.shape}')
      
      ''' method about calculation of distances between images
      dist_list = []
      for idx, emb_db in enumerate(embedding_list):
        dist = calculateDistance(emb, emb_db)
        dist_list.append(round(dist,4))

      # min_dist
      min_dist = min(dist_list) # get minumum dist value
      min_dist_idx = dist_list.index(min_dist) # get minumum dist index
      min_name = name_list[min_dist_idx] # get matched_name corrosponding to minimum dist

      if min_dist < self.face_dist_threshold:
        print('='*100)
        print('matched: ', min_name)
        print('name_list: ',name_list)
        print('dist_list: ',dist_list)
        print('='*100)

        text='['+min_name+'] '+str(round(min_dist,4))
        image = cv2.putText(image, text, (absx, absy-40), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (0,255,0),3, cv2.LINE_AA)
      '''
      return image

  def detectFaces(self, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    # To improve performance, optionally mark the frame as not writeable to
    # pass by reference.
    frame.flags.writeable = False

    # FACE DETECTION USING MEDIAPIPE
    results = self.face_detection.process(frame)

    # Draw the face detection annotations on the frame.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame, results, h, w

  def checkCenterOfFace(self, center_of_face, h, w):
    x_face_center = center_of_face[0]
    y_face_center = center_of_face[1]

    x1 = int(w/2 - self.center_area_size_half)
    y1 = int(h/2 - self.center_area_size_half)
    x2 = int(w/2 + self.center_area_size_half)
    y2 = int(h/2 + self.center_area_size_half)
    # print('center: ', x_face_center, y_face_center)
    # print('box: ', x1, x2, y1, y2)

    if x_face_center > x1 and x_face_center < x2 and y_face_center > y1 and y_face_center < y2:
      return True
    else:
      return False
    
  def ConvertFromDetectionToCoordinate(self, detection):
    location = detection.location_data
    relative_bounding_box = location.relative_bounding_box
    x_min = relative_bounding_box.xmin
    y_min = relative_bounding_box.ymin
    widthh = relative_bounding_box.width
    heightt = relative_bounding_box.height

    return x_min, y_min, widthh, heightt

  def alignCenters(self, frame):
    _, results, h, w =  self.detectFaces(frame)
    iscenter = False
    # print(type(results.detections), results.detections)

    if len(results.detections) > 1:
      print("Too many Faces are detected")
      return frame, iscenter
    else:
      detection = results.detections[0]
      x_min, y_min, widthh, heightt = self.ConvertFromDetectionToCoordinate(detection)
      absx, absy=self.mp_drawing._normalized_to_pixel_coordinates(x_min, y_min, w, h)
      abswidth, absheight = self.mp_drawing._normalized_to_pixel_coordinates(x_min+widthh,y_min+heightt,w,h) 
      center_of_face = (int(absx + abswidth)//2, int(absy + absheight)//2)

      # print(absx,absy,abswidth, absheight, center_of_face)

      cv2.circle(frame, center_of_face , radius=5, color=(255, 0, 0), thickness=3)
      iscenter = self.checkCenterOfFace(center_of_face, h, w)

      return frame, iscenter
      
  def execute_face_application(self, frame, embedding_data=None, HeadPoseEstimation=False, FaceIdentification=False):

    if embedding_data is None and FaceIdentification is True:
      raise Exception("There is no embedding data")

    org_image = frame.copy()
    image, results, h, w = self.detectFaces(frame)

    # IF FACE IS DETECTED, HEAD POSE ESTIMATION IS PROCEEDED
    if results.detections:
      for detection in results.detections:
        x_min, y_min, widthh, heightt = self.ConvertFromDetectionToCoordinate(detection)

        # if there is None return, continue this loop
        try:
          absx, absy=self.mp_drawing._normalized_to_pixel_coordinates(x_min, y_min, w, h)
          abswidth, absheight = self.mp_drawing._normalized_to_pixel_coordinates(x_min+widthh,y_min+heightt,w,h) 
        except:
          continue

        # HEAD POSE ESTIMATION 
        if HeadPoseEstimation:
          image = self.estimateHeadPose(org_image, image, absx, absy, abswidth, absheight)
        
        # FACE IDENTIFICATION
        if FaceIdentification and embedding_data is not None:
          image = self.identifyFace(embedding_data, org_image, image, absx, absy, abswidth, absheight)
    
    return image

  def show_instruction(self, frame):
    (h, w) = frame.shape[:2]
    background_image = np.zeros((h, w, 3), dtype="uint8")
    coordinate = (int(w/4), int(h/2))

    instruction_text = "First, position your face in the camera frame. "
    instruction_image = cv2.putText(background_image, instruction_text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 
                      fontScale=0.7, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    instruction_text = "Then, move your head in a circle to show all the angles of your face."
    coordinate = (int(w/4), int(h/2+20))
    instruction_image = cv2.putText(background_image, instruction_text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 
                  fontScale=0.7, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("mac", instruction_image)




