from turtle import back
import cv2
import numpy as np


class FaceRegisterer():
  def __init__(self):
    self.isntCompleted = True
    self.newUser = ''
    self.w = 0
    self.h = 0

  def show_instruction(self, frame):
    (self.h, self.w) = frame.shape[:2]
    background_image = np.zeros((self.h, self.w, 3), dtype="uint8")
    coordinate = (int(self.w/4), int(self.h/2))

    instruction_text = "First, position your face in the camera frame. "
    instruction_image = cv2.putText(background_image, instruction_text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 
                      fontScale=0.7, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    instruction_text = "Then, move your head in a circle to show all the angles of your face."
    coordinate = (int(self.w/4), int(self.h/2+20))
    instruction_image = cv2.putText(background_image, instruction_text, coordinate, cv2.FONT_HERSHEY_SIMPLEX, 
                  fontScale=0.7, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("mac", instruction_image)
  