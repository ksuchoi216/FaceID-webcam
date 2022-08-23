import os
import cv2

from modules.utils import draw_center_border, saveFrame


# def captureFrames(args, dataAdministrator, faceAnalyst, faceRegisterer, decisionMaker=None, objectTracker=None):
def captureFrames(config, dataAdministrator, faceAnalyst, decisionMaker=None):
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

    if key == 116:
      saveFrame(frame)

    if key == 114:
      isregistration = True
      faceAnalyst.show_instruction(frame)
      cv2.waitKey(100)

    if isregistration == True:
      # print('Starting registration...')
      frame = draw_center_border(frame, config["center_area_size_half"], (127, 255, 255), 5, 5, 10)
      frame, iscenter = faceAnalyst.alignCenters(frame)
      if iscenter is True:
        dataAdministrator.createNewDirectory()
        isregistration = dataAdministrator.saveFrameInFolder(org_frame)

    # frame = faceAnalyst.detectFaces(frame, isregistration, embedding_data=dataAdministrator.get_embedding_data(), HeadPoseEstimation=True, FaceIdentification=True)
    
    # # object detection
    # frame = objectTracker.detect_objects(frame)

    # # FACE DETECTION
    # detected_frame = faceAnalyst.match_faces(frame, dataAdministrator.get_embedding_data())
    # if detected_frame is not None:
    #   frame = detected_frame


    cv2.imshow("mac", frame)
  
  # vc.release()
  # cv2.destroyWindow("mac")