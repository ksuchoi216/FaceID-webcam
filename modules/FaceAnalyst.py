import os
import sys

import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import dlib
import mediapipe as mp

from .external_library import MTCNN, InceptionResnetV1
from .external_library.HeadPoseEstimation.drawFace import draw
from .external_library.HeadPoseEstimation import reference_world as world
from .external_library.sort.sort import Sort


def calculateDistance(A, B):
    return torch.dist(A, B).item()


def calculateSimilarity(A, B):
    return F.cosine_similarity(A, B).item()


class FaceAnalyst:
    def __init__(self, cfg):
        """_summary_

        Args:
            sscfg (dict): "config file only for FaceAnalyst
                    such as config["FaceAnlyst"]
        """

        IsHeadPoseEstimation = cfg["Options"]["IsHeadPoseEstimation"]
        IsFaceIdentification = cfg["Options"]["IsFaceIdentification"]
        IsObjectTracking = cfg["Options"]["IsObjectTracking"]
        IsEyeTracking = cfg["Options"]["IsEyeTracking"]
        IsVideoTest = cfg["Options"]["IsVideoTest"]
        path_for_video = cfg["Options"]["path_for_video"]

        self.correct_y_range = cfg["Options"]["correct_y_range"]
        self.registered_users = cfg["registered_users"]
        num_classes = len(self.registered_users)
        cfg_m = cfg["model"]

        customed_pretrained_model = cfg_m["customed_pretrained_model"]
        image_size_for_face_detector = cfg_m["image_size_for_face_detector"]
        path_for_pretrained_model = cfg_m["path_for_pretrained_model"]
        device = torch.device("cpu")

        # facenet models to be loaded
        self.single_face_detector = MTCNN(
            image_size=image_size_for_face_detector,
            margin=0,
            keep_all=False,
            min_face_size=40,
        )  # keep_all=False
        self.multi_faces_detector = MTCNN(
            image_size=image_size_for_face_detector,
            margin=0,
            keep_all=True,
            min_face_size=40,
        )  # keep_all=True
        self.face_feature_extractor = InceptionResnetV1(
            pretrained=cfg_m["pretrained"],
            classify=cfg_m["classify"],
            num_classes=num_classes,
            device=device,
        )

        # trained model to be loaded
        if customed_pretrained_model is True:
            self.face_feature_extractor.load_state_dict(
                torch.load(path_for_pretrained_model, map_location=device)
            )
            self.face_feature_extractor.classify = True
        self.face_feature_extractor.eval()

        # threshold values for filtering face images
        self.face_prob_threshold1 = cfg["face_prob_threshold1"]
        self.face_prob_threshold2 = cfg["face_prob_threshold2"]
        self.face_dist_threshold = cfg["face_dist_threshold"]

        # normalisation and resize face images with interpolation
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # related to face detection using mediapipe
        self.focal = cfg["focal"]
        self.mp_face_detection = (
            mp.solutions.face_detection
        )  # face detection by using mediapipe
        self.mp_drawing = (
            mp.solutions.drawing_utils
        )  # drawing face landmarks by using mediapipe
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=cfg['min_detection_confidence']
        )

        # related to head pose estimation
        PREDICTOR_PATH = os.path.join(
            "./modules/external_library/HeadPoseEstimation/models/",
            "shape_predictor_68_face_landmarks.dat",
        )
        if not os.path.isfile(PREDICTOR_PATH):
            print("PREDICTOR_PATH: ", PREDICTOR_PATH)
            print("[ERROR] USE models/downloader.sh to download the predictor")
            sys.exit()
        self.predictor = dlib.shape_predictor(
            PREDICTOR_PATH
        )  # trained model for predicting human pose.
        self.face3Dmodel = world.ref3DModel()

        # related to Object tracking
        self.object_detector = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", pretrained=True
        )  # model to be loaded for detecting objects (named 'YOLO version 5')
        self.object_detector.float()
        self.object_detector.eval()
        self.mot_tracker = (
            Sort()
        )  # model to be loaded for tracking object(multiple objects tracking)

        self.center_area_size_half = cfg["center_area_size_half"]

        # ===========================================
        self.ID_card_num = cfg['ID_card_num']

        # ===========================================
        cfg_dist = cfg["distance_mode"]
        self.IsDistanceMode = cfg_dist['IsDistanceMode']
        if self.IsDistanceMode:
            self.IsCosSimilarity = cfg_dist['IsCosSimilarity']
            path_ = cfg_dist['filepath_photos']
            print(f'loading registered photos from {path_}')
            dataset = datasets.ImageFolder(path_)
            idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

            def collate_fn(x):
                return x[0]

            dataloader = DataLoader(dataset, collate_fn=collate_fn)
            self.name_list = []
            self.embedding_list = []

            for img, idx in dataloader:
                face, prob = self.single_face_detector(img, return_prob=True)
                if face is not None and prob > self.face_prob_threshold1:
                    emb = self.face_feature_extractor(face.unsqueeze(0))
                    self.embedding_list.append(emb.detach())
                    self.name_list.append(idx_to_class[idx])

    def get_single_face_detector(self):
        return self.single_face_detector

    # ========================================================================================================================
    # [3] FACE IDENTIFICATION - FUNCTION
    # ------------------------------------------------------------------------------------------------------------------------
    def identifyFace(
        self, org_image, image,
        start_point,
        end_point,
        IsDrawing
    ):
        """
        identification of a detected face taken by the mediapipe face detector

        Args:
            org_image (BGR image): original image for detecting a face
            and extracting features from a face.
            image (BGR image): an image drawn with face detection,
            head pose estimation, and object tracking
            abs_min_x (float): absolute min x coordinate
            abs_min_y (float): absolute min y coordinate
            abs_max_x (float): absolute max x coordinate
            abs_max_y (float): absolute max y coordinate


        Returns:
            image (BGR image): an image drawn with information
            regarding the recognized face
        """

        abs_min_x, abs_min_y = start_point
        abs_max_x, abs_max_y = end_point

        cropped_face_image = org_image[abs_min_y:abs_max_y,
                                       abs_min_x:abs_max_x]
        cropped_face_image = Image.fromarray(cropped_face_image)
        # cropped_face_image = self.transform(cropped_face_image)

        # added process about detecting faces for filtering faces
        # and matching pytorch form
        face, prob = self.single_face_detector(cropped_face_image,
                                               return_prob=True)

        # method about calculation of distances between images
        if face is not None and prob > self.face_prob_threshold1:
            # MTCNN face detection
            boxes, _ = self.single_face_detector.detect(org_image)
            draw = ImageDraw.Draw(image)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(0, 0, 255), width=6)

            if self.IsDistanceMode:
                emb = self.face_feature_extractor(face.unsqueeze(0)).detach()
                dist_list = []

                for idx, emb_db in enumerate(self.embedding_list):
                    if self.IsCosSimilarity:
                        dist = calculateSimilarity(emb, emb_db)
                    else:
                        dist = calculateDistance(emb, emb_db)

                    dist_list.append(round(dist, 3))

                if self.IsCosSimilarity:
                    target_dist = max(dist_list)
                else:
                    target_dist = min(dist_list)

                print(f'dist_list result: {dist_list}')

                target_idx = dist_list.index(target_dist)
                name = self.name_list[target_idx]
                res = target_dist

            else:
                results = self.face_feature_extractor(face.unsqueeze(0))
                results = torch.sigmoid(results)
                print(f'results: {results}')
                # prob, index = torch.max(results, 1)
                probs, indexs = torch.topk(results, k=2, dim=1)
                prob = probs[0][0]
                index = indexs[0][0]
                res = prob
                # diff = probs[0][0]-probs[0][1]
                name = self.registered_users[index]

            if prob > self.face_prob_threshold2:
                # print(self.registered_users[index], prob)
                # rect for Mediapipe face detection
                cv2.rectangle(
                    image,
                    (abs_min_x, abs_min_y),
                    (abs_max_x, abs_max_y),
                    (255, 0, 0),
                    2,
                )

                text_for_faceid = (
                    f"(name, prob) = "
                    f"({name}, {res:.3})"
                )
                cv2.putText(
                    image, text_for_faceid,
                    (int(abs_min_x)-30, int(abs_min_y-40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (226, 181, 0), 2
                )

                # rect for MTCNN face detection
                # cv2.rectangle(
                #     image,
                #     (int(box[0]), int(box[1])),
                #     (int(box[2]), int(box[3])),
                #     (0, 255, 80),
                #     2,
                # )

        return image

    # ========================================================================================================================

    # ========================================================================================================================
    # [2] HEAD POSE ESTIMATION - FUNCTION
    # ------------------------------------------------------------------------------------------------------------------------
    def estimateHeadPose(
        self,
        org_image,
        image,
        start_point,
        end_point,
        CalculateAngle=False,
        Calculate3Dcoordinates=False,
    ):
        """calculation for the angle of detected face's head
        Args:
            org_image (BGR image): BGR image taken by OpenCV
            image (BGR image): ditto
            abs_min_x (float): absolute min x coordinate
            abs_min_y (float): absolute min y coordinate
            abs_max_x (float): absolute max x coordinate
            abs_max_y (float): absolute max y coordinate

        Returns:
            image : BGR image drawn by OpenCV
        """
        abs_min_x, abs_min_y = start_point
        abs_max_x, abs_max_y = end_point

        img = org_image
        new_rect = dlib.rectangle(abs_min_x, abs_min_y, abs_max_x, abs_max_y)
        shape = self.predictor(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB), new_rect
        )  # shape is outline of detected face

        draw(image, shape)  # draw the ouline of detected face

        refImgPts = world.ref2dImagePoints(
            shape
        )  # get reference points in 2D (6 points)

        height, width, channels = img.shape
        # focal is distance between object and camera
        focalLength = self.focal * width
        cameraMatrix = world.cameraMatrix(focalLength, (height / 2, width / 2))

        mdists = np.zeros((4, 1), dtype=np.float64)

        # calculate rotation and translation vector using solvePnP
        # refer to https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
        success, rotationVector, translationVector = cv2.solvePnP(
            self.face3Dmodel, refImgPts, cameraMatrix, mdists
        )

        noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        noseEndPoint2D, jacobian = cv2.projectPoints(
            noseEndPoints3D, rotationVector,
            translationVector, cameraMatrix, mdists
        )

        #  draw nose line
        p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
        p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
        cv2.line(image, p1, p2, (110, 220, 0),
                 thickness=2, lineType=cv2.LINE_AA)

        # calculating euler angles
        # This can be used to transform all three basis vectors
        #   to compute a rotation matrix
        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        # print('*' * 80)
        # print(f"Qx:{Qx}\tQy:{Qy}\tQz:{Qz}\t")

        if Calculate3Dcoordinates:
            x = np.arctan2(Qx[2][1], Qx[2][2])
            y = np.arctan2(
                -Qy[2][0],
                np.sqrt((Qy[2][1] * Qy[2][1]) + (Qy[2][2] * Qy[2][2]))
            )
            z = np.arctan2(Qz[0][0], Qz[1][0])
            text = f"predicted 3D: ({x:.2f},{y:.2f},{z:.2f})"
            cv2.putText(
                image,
                text,
                (abs_max_x, abs_max_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (2, 255, 80),
                2,
            )

        if CalculateAngle:
            if angles[1] < -20:
                GAZE = "Looking: Left"
            elif angles[1] > 20:
                GAZE = "Looking: Right"
            else:
                GAZE = "Unknown direction"

            cv2.putText(
                image,
                GAZE,
                (abs_min_x, abs_max_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 80),
                2,
            )

        return image

    # ========================================================================================================================

    # ========================================================================================================================
    # [1] OBJECT TRACKING - FUNCTION
    # ------------------------------------------------------------------------------------------------------------------------
    def track_objects(self, image, ID_cards, IsDrawing=True):
        """The order of execution for tracking objects is presented as follows:
        1. detection of objects
        2. filtering objects to get only humans
        3. tracking the filtered objects(humans)

        Args:
            image (BGR image): the BGR image taken by OpenCV

        Returns:
            image (BGR image): the BGR image drawn with object detection
        """

        results = self.object_detector(image)
        df = results.pandas().xyxy[0]
        detections = df[df["name"] == "person"].drop(columns="name").to_numpy()
        track_ids = self.mot_tracker.update(detections)

        for i in range(len(track_ids.tolist())):
            coords = track_ids.tolist()[i]
            xmin, ymin, xmax, ymax = (
                int(coords[0]),
                int(coords[1]),
                int(coords[2]),
                int(coords[3]),
            )
            name_idx = int(coords[4])
            ID_cards[name_idx] = []
            name = "ID: {}".format(str(name_idx))

            if IsDrawing:
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax),
                                      (255, 0, 0), 2)
                image = cv2.putText(
                    image,
                    name,
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

        return image

    # ========================================================================================================================

    # ========================================================================================================================
    # [4] EYES TRACKING - FUNCTION
    # ------------------------------------------------------------------------------------------------------------------------
    def track_eyes(self, org_image, image,
                   eyeTracker,
                   start_point,
                   end_point,
                   IsDrawing=True):

        image = eyeTracker.compute_diff(
            org_image, image, start_point, end_point, IsDrawing)

        return image

    # ========================================================================================================================

    def detectFaces(self, frame):
        """
        face detection with converting color from BGR to RGB, and if this
        process is done, convert image's color to BGR color

        Args:
            frame (BGR image): BGR image extracted by OpenCV video capturing

        Returns:
            frame (BGR image): BGR image
            results (list) : detected faces
            h (float): height of frame image
            w (float): width of frame image
        """

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        # To improve performance, optionally mark the frame as not writeable to
        # pass by reference.
        frame.flags.writeable = False

        # FACE DETECTION USING MEDIAPIPE
        results = self.face_detection.process(frame)

        # Draw the face detection annotations on the frame.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame, results, h, w

    def ConvertToCoordinate(self, detection):
        """get coordinates from detected face by mediapipe

        Args:
            detection (list): detected faces list by mediapipe

        Returns:
            x_min (float) : min x coordinate
            y_min (float) : min y coordinate
            x_max (float) : max x coordinate (add together min x and width)
            y_max (float) : max y coordinate (add together min y and height)
        """
        location = detection.location_data
        relative_bounding_box = location.relative_bounding_box
        x_min = relative_bounding_box.xmin
        y_min = relative_bounding_box.ymin
        x_max = relative_bounding_box.width
        y_max = relative_bounding_box.height

        return x_min, y_min, x_max, y_max

    def draw_basic_info(self, image, start_point, end_point, IsDrawing=True):
        abs_x_min, abs_y_min = start_point
        abs_x_max, abs_y_max = end_point

        self.face_center = (((abs_x_min+abs_x_max)//2),
                            ((abs_y_min+abs_y_max)//2))
        # center for face
        cv2.circle(image, self.face_center, 0, (0, 0, 255), 7)

        center_line = self.width//2
        self.IsLeftSeat = self.face_center[0] < center_line
        # right = self.face_center[0] > (self.width//2)

        if IsDrawing:
            # if self.IsLeftSeat:
            #     cv2.putText(
            #         image, '[1]', (self.width//4, 40),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.9, (0, 255, 0), 3
            #     )
            #     cv2.rectangle(
            #         image,
            #         (0, 0),
            #         (center_line, self.height),
            #         (0, 255, 0), 6
            #     )
            # else:
            #     cv2.putText(
            #         image, '[0]', ((self.width//4)*3, 40),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.9, (255, 0, 0), 3
            #     )
            #     cv2.rectangle(
            #         image,
            #         (center_line, 0),
            #         (self.width, self.height),
            #         (0, 255, 0), 6
            #     )

            # rectangle for face
            cv2.rectangle(
                image,
                (abs_x_min, abs_y_min),
                (abs_x_max, abs_y_max),
                (0, 255, 0), 3
            )
        return image

    def execute_face_application(
        self,
        frame,
        HeadPoseEstimation=False,
        FaceIdentification=False,
        ID_cards=None,
        ObjectTracking=False,
        EyeTracking=True,
        eyeTracker=None,
        IsDrawing=True
    ):
        """
        Args:
            frame (BGR image): an extracted frame image from webcam
              by OpenCV which based on BGR color
            HeadPoseEstimation (bool, optional): selection for head pose
              estimation(calculating the face angle). Defaults to False.
            FaceIdentification (bool, optional): selection for identifying a
              face from registered faces which trained the classifier
              in advance. Defaults to False.
            ObjectTracking (bool, optional): selection for tracking human
              including object detection. Defaults to False.

        Returns:
            frame (BGR image) with drawing options: frame image drawn together
            with selected options.
        """
        self.height, self.width, _ = frame.shape

        org_image = frame.copy()
        image, results, h, w = self.detectFaces(frame)

        # If multiple faces are detected, length of detected faces list is
        #   more than two. Therefore, "for loop" is used
        #   for processing each face.
        if results.detections:
            for detection in results.detections:
                x_min, y_min, x_max, y_max = self.ConvertToCoordinate(
                    detection
                )

                # if there is None return, continue this loop
                try:
                    (
                        abs_x_min,
                        abs_y_min,
                    ) = self.mp_drawing._normalized_to_pixel_coordinates(
                        x_min, y_min, w, h
                    )
                    (
                        abs_x_max,
                        abs_y_max,
                    ) = self.mp_drawing._normalized_to_pixel_coordinates(
                        x_min + x_max, y_min + y_max, w, h
                    )
                except BaseException:
                    continue

                image = self.draw_basic_info(
                    image,
                    [abs_x_min, abs_y_min],
                    [abs_x_max, abs_y_max],
                    IsDrawing=IsDrawing["basic_info"]
                )

                # [1] OBJECT TRACKING
                if ObjectTracking:
                    image = self.track_objects(image, ID_cards)

                # [2] HEAD POSE ESTIMATION
                if HeadPoseEstimation:
                    image = self.estimateHeadPose(
                        org_image,
                        image,
                        [abs_x_min, abs_y_min],
                        [abs_x_max, abs_y_max],
                        CalculateAngle=True,
                        Calculate3Dcoordinates=True
                    )

                # [3] FACE IDENTIFICATION
                if FaceIdentification:
                    image = self.identifyFace(
                        org_image, image,
                        [abs_x_min, abs_y_min],
                        [abs_x_max, abs_y_max],
                        IsDrawing
                    )

                # [4] EYE TRACKING
                if EyeTracking:
                    image = self.track_eyes(
                        org_image, image,
                        eyeTracker,
                        [abs_x_min, abs_y_min],
                        [abs_x_max, abs_y_max]
                    )

        return image
