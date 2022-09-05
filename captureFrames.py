import os
import cv2

from modules.utils import draw_center_border, saveFrame

def captureFrames(config, faceAnalyst):
  """_summary_

  Args:
      config (dict): config named "default" config such as config["default"]
      faceAnalyst (class): this object for analysing images 
                such as object tracking, head pose estimation, face detection, and face recognition.

  Raises:
      Exception: if there is no captured image, raise the exception
      
  """
  print('\nstart video capturing')

  frame_width = config["frame_width"]
  frame_height = config["frame_height"]

  vc = cv2.VideoCapture(0)

  if not vc.isOpened():
    raise Exception("Could not open video capture")
  else:
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
  
  isregistration = config["isregistration"] # default is False
  count = 0
  saved_frame_count = 0

  while True:
    ret, frame = vc.read()
    org_frame = frame.copy()
    # print(type(frame), frame.shape)

    # check whether a frame is captured or not
    if not ret:
      print("failed to capture frames")
      break

    # wait key ESC(no. 27) to break the while loop
    key = cv2.waitKey(1)
    if key == 27:
      vc.release()
      cv2.destroyWindow("mac")
      break
    
    # frame = faceAnalyst.execute_face_application(frame, embedding_data=dataAdministrator.get_embedding_data(), HeadPoseEstimation=True, FaceIdentification=True)
    frame = faceAnalyst.execute_face_application(frame, HeadPoseEstimation=True, FaceIdentification=True, ObjectTracking=True)

    cv2.imshow("mac", frame)
  
  # vc.release()
  # cv2.destroyWindow("mac")