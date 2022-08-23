import torch
import cv2
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tkinter import *

def print_dict(dict):
  for k, v in dict.items():
    print(k,': ',v)

def printd(img):
  print(img.shape, type(img))

def drawHist(img, tensor=False):
  if tensor == True:
    img = img.numpy()

  plt.hist(img.ravel(), bins=50, density=True);
  plt.xlabel("pixel values")
  plt.ylabel("relative frequency")
  plt.title("distribution of pixels");
  plt.show()

def draw_center_border(frame, center_area_size_half, color, thickness, r, d):
  (h, w) = frame.shape[:2]

  center_x = int(w/2)
  center_y = int(h/2)
  x1 = int(w/2 - center_area_size_half)
  y1 = int(h/2 - center_area_size_half)
  x2 = int(w/2 + center_area_size_half)
  y2 = int(h/2 + center_area_size_half)

  cv2.circle(frame, (center_x, center_y), 2, (0,0,255), thickness)

  # Top left
  cv2.line(frame, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
  cv2.line(frame, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
  cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
  # Top right
  cv2.line(frame, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
  cv2.line(frame, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
  cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
  # Bottom left
  cv2.line(frame, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
  cv2.line(frame, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
  cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
  # Bottom right
  cv2.line(frame, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
  cv2.line(frame, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
  cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

  return frame

def saveFrame(frame):
  # im = Image.fromarray(frame).convert("RGB")
  # im.save('test.png')
  cv2.imwrite('test.png', frame)
  print("test.png was saved successfully")
  
class TextBox():
  def __init__(self):
    self.user_name = ''

    # Creating the tkinter Window
    self.root = Tk()
    self.root.geometry("300x100")
    self.root.title('User name input window')

    self.e = Entry(self.root)
    self.e.pack()
    self.e.focus_set()
    
    # Button for closing
    exit_button = Button(self.root, text="Complete", command=self.Close)
    exit_button.pack(pady=20)

    self.root.mainloop()

  # Function for closing window
  def Close(self):
    self.user_name = self.e.get()
    self.root.destroy()

    # return self.user_name

