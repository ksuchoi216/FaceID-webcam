import cv2

from modules import FaceAnalyst, EyeTracker


def captureFrames(config):
    """
    Args:
        config (dictctrl y): config named "default"
        config such as config["default"]
        faceAnalyst (class): this object for analysing images
                  such as object tracking, head pose estimation,
                  face detection, and face recognition.

    Raises:
        Exception: if there is no captured image, raise the exception

    """
    print("Start video capturing...")
    IsDrawing = config["IsDrawing"]
    IsHeadPoseEstimation = config["Options"]["IsHeadPoseEstimation"]
    IsFaceIdentification = config["Options"]["IsFaceIdentification"]
    IsObjectTracking = config["Options"]["IsObjectTracking"]
    IsEyeTracking = config["Options"]["IsEyeTracking"]
    correct_y_range = config["Options"]["correct_y_range"]

    # frame_width = config["frame_width"]
    # frame_height = config["frame_height"]

    vc = cv2.VideoCapture(0)

    # if not vc.isOpened():
    #     raise Exception("Could not open video capture")
    # else:
    #     vc.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    #     vc.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    ret, frame = vc.read()
    height, width, _ = frame.shape

    # faceAnalyst
    config["FaceAnalyst"]["correct_y_range"] = correct_y_range
    faceAnalyst = FaceAnalyst(config["FaceAnalyst"])

    # EyeTracker
    config["EyeTracker"]["correct_y_range"] = correct_y_range
    eyeTracker = EyeTracker(config["EyeTracker"],
                            faceAnalyst.get_single_face_detector(),
                            frame)

    ID_cards = {}
    while True:
        ret, frame = vc.read()
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

        if IsDrawing['EyeTracking'] and IsEyeTracking:
            cv2.rectangle(frame, (0, correct_y_range[0]),
                          (width, correct_y_range[1]),
                          (0, 0, 255), 3)

        frame = faceAnalyst.execute_face_application(
            frame,
            HeadPoseEstimation=IsHeadPoseEstimation,
            FaceIdentification=IsFaceIdentification,
            ID_cards=ID_cards,
            ObjectTracking=IsObjectTracking,
            EyeTracking=IsEyeTracking,
            eyeTracker=eyeTracker,
            IsDrawing=IsDrawing
        )

        cv2.imshow("mac", frame)
