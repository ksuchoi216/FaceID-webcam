import os

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

from modules.external_library import MTCNN, InceptionResnetV1

mtcnn_detect = MTCNN(
    image_size=240,
    margin=0,
    keep_all=False,
    min_face_size=40,
)
mtcnn_show = MTCNN(select_largest=False, post_process=False)
mp_face_detection = (
    mp.solutions.face_detection
)  # face detection by using mediapipe
mp_drawing = (
    mp.solutions.drawing_utils
)  # drawing face landmarks by using mediapipe
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
)


def detectFaces(frame):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    # To improve performance, optionally mark the frame as not writeable to
    # pass by reference.
    frame.flags.writeable = False

    # FACE DETECTION USING MEDIAPIPE
    results = face_detection.process(frame)

    # Draw the face detection annotations on the frame.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame, results, h, w


def ConvertToCoordinate(detection):

    location = detection.location_data
    relative_bounding_box = location.relative_bounding_box
    x_min = relative_bounding_box.xmin
    y_min = relative_bounding_box.ymin
    x_max = relative_bounding_box.width
    y_max = relative_bounding_box.height

    return x_min, y_min, x_max, y_max


def detect_face(frame, isBreak):
    print('detect_face start')
    height, width, _ = frame.shape

    image, results, h, w = detectFaces(frame)

    if results.detections:
        for detection in results.detections:
            x_min, y_min, x_max, y_max = ConvertToCoordinate(
                detection
            )
            # if there is None return, continue this loop
            try:
                (
                    abs_x_min,
                    abs_y_min,
                ) = mp_drawing._normalized_to_pixel_coordinates(
                    x_min, y_min, w, h
                )
                (
                    abs_x_max,
                    abs_y_max,
                ) = mp_drawing._normalized_to_pixel_coordinates(
                    x_min + x_max, y_min + y_max, w, h
                )
            except BaseException:
                continue

            image_face = image[abs_y_min:abs_y_max, abs_x_min:abs_x_max, :]
            print(f'detected face shape: {image_face.shape}')

            cv2.rectangle(
                image,
                (abs_x_min, abs_y_min),
                (abs_x_max, abs_y_max),
                (0, 255, 0), 3
            )

            isBreak = True
            return image, isBreak

    return image, isBreak


def captureVideo():
    source_path = './data/video_resolution'
    device_names = ['macbook', 'android']
    extension_names = ['.mov', '.mp4']

    for device_name, extension_name in zip(device_names,
                                           extension_names):
        file_name = 'video_resolution_'+device_name+extension_name
        path_ = os.path.join(source_path, file_name)
        print(f'[{device_name}]')
        print(path_)

        vc = cv2.VideoCapture(path_)

        isBreak = False
        while True:
            success, frame = vc.read()
            print(f'frame shape: {frame.shape}')

            if success:
                image, isBreak = detect_face(frame, isBreak)

                if isBreak:
                    print('Done')
                    plt.imshow(image)
                    plt.show()
                    break


if __name__ == '__main__':
    captureVideo()
