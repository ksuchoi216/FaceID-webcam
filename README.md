# Installation

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


# package
pip install mediapipe
pip install dlib