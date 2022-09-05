# INSTALLATION
## facenet
git clone git@github.com:timesler/facenet-pytorch.git ./modules/external_library/facenet_pytorch/

## head pose estimation
git clone git@github.com:by-sabbir/HeadPoseEstimation.git ./modules/external_library/HeadPoseEstimation/

### to download model
cd ./modules/external_library/HeadPoseEstimation/models/
bash downloader.sh
cd ../../../..

<!-- ## mediapipe -->
<!-- git clone https://github.com/google/mediapipe.git  -->

## important packages
pip install mediapipe
pip install dlib

# CODE STRUCTURE
This code is inclueded in main.py, captureFrames.py, and additional modules.
The flow of the code's execution is main.py, captureFrames.py

## Explanation for each files
main.py is for parsing arguments and declaration about related objects such as FaceAnalyst.

captureFrames.py is for receiving frames from webcam by utilsing OpenCV.

FaceAnalyst.py is for face detection, head pose estimation, and object tracking.

The following is order of execution for this code.
1. main.py
  - main() needs config(including parsed arguments)
    - captureFrames() needs the object which is FaceAnalyst.
2. captureFrames.py
  - cv2.VideoCapture captures each frames from webcam. 
  - If capturing a frame is successed, "while loop" continues to take each frame as a image.
  - If you want to stop capturing images, please press ESC key (ASCII NO. 27)
  - Note) OpenCV image is based on "BGR" color. 
    - faceAnalyst.execute_face_application() needs the following parameters.
      - HeadPoseEstimation: option selection for calculating an angle of detected faces
      - FaceIdentification: option selection for recognizing a face from registered faces.
      - ObjectTracking: optino selection for tracking the "human" object
  - if you want to get further information regarding faceAnalyst.execute_face_application(), please refer to docstrings in the code.

# REFERENCES

## facenet tensorflow
https://github.com/davidsandberg/facenet
## facenet pytorch
https://github.com/davidsandberg/facenet

## yolov5 pytorch
https://pytorch.org/hub/ultralytics_yolov5/
