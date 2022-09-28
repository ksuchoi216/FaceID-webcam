#!/bin/bash
repo1="git@github.com:timesler/facenet-pytorch.git"
folder1="./modules/external_library/facenet_pytorch/"
repo2="git@github.com:by-sabbir/HeadPoseEstimation.git"
folder2="./modules/external_library/HeadPoseEstimation/"


git clone "$repo1" "$folder1"
git clone "$repo2" "$folder2"

cd ./modules/external_library/HeadPoseEstimation/models/
bash downloader.sh
cd ../../../..


wget   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 # DOWNLOAD LINK
bunzip2 ./shape_predictor_68_face_landmarks.dat.bz2