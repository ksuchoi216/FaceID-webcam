import cv2


def captureFrames(config, faceAnalyst):
    """
    Args:
        config (dict): config named "default" config such as config["default"]
        faceAnalyst (class): this object for analysing images
                  such as object tracking, head pose estimation,
                  face detection, and face recognition.

    Raises:
        Exception: if there is no captured image, raise the exception

    """
    print("\nstart video capturing")

    frame_width = config["frame_width"]
    frame_height = config["frame_height"]

    vc = cv2.VideoCapture(0)

    if not vc.isOpened():
        raise Exception("Could not open video capture")
    else:
        vc.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        vc.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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

        frame = faceAnalyst.execute_face_application(
            frame,
            HeadPoseEstimation=False,
            FaceIdentification=False,
            ObjectTracking=False,
            EyeTracking=True,
        )

        cv2.imshow("mac", frame)