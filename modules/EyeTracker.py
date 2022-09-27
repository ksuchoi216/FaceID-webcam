import os

import numpy as np
from PIL import Image
import cv2
import dlib


class EyeTracker(object):
    def __init__(self, cfg, single_face_detector, frame):
        h, w, _ = frame.shape
        self.width = w
        print(f'frame dimension: {h}, {w}')
        self.correct_y_range = np.array(cfg['correct_y_range'])
        self.correct_coords = np.array([np.mean(self.correct_y_range,
                                                dtype=np.int16),
                                        w//2])
        print(f'correct_coords: {self.correct_coords}')
        
        folder_for_preset_images = cfg['folder_for_preset_images']
        face_center_list = []
        for file_name in cfg['preset_image_name_list']:
            path = os.path.join(folder_for_preset_images, file_name)
            im = Image.open(path)
            boxes, _ = single_face_detector.detect(im)
            box = boxes[0]
            face_center = np.array([(box[0]+box[2])//2, (box[1]+box[3])//2])
            face_center_list.append(face_center)

        face_center_list = np.array(face_center_list)
        print(f'face_center_list: {face_center_list}')

        first_move_setting = np.asarray(cfg['first_move_setting'])
        second_move_setting = np.asarray(cfg['second_move_setting'])

        move_mat = np.stack((first_move_setting, second_move_setting), axis=-1)

        delta_vec1 = face_center_list[1] - face_center_list[0]
        delta_vec2 = face_center_list[2] - face_center_list[1]

        delta_mat = np.stack((delta_vec1, delta_vec2), axis=-1)

        delta_mat_inv = np.linalg.inv(delta_mat)
        print('check inverse matrix')
        print(np.allclose(np.dot(delta_mat, delta_mat_inv), np.eye(2)))
        print(np.allclose(np.dot(delta_mat_inv, delta_mat), np.eye(2)))

        self.para_mat = np.matmul(move_mat, delta_mat_inv)
        
        print(f'parameters matrix was calculated: \n {self.para_mat}')

        self.correct_pred_coords = np.matmul(self.para_mat,
                                             self.correct_coords)

        # =====================================================================
        self.landmark_predictor = dlib.shape_predictor(
            './shape_predictor_68_face_landmarks.dat'
            )

        left = [36, 37, 38, 39, 40, 41]
        right = [42, 43, 44, 45, 46, 47]
        self.eyes_nums = left + right

    def get_diff_from_eyes_center(self, eyes_center):
        eyes_pred_coords = np.matmul(self.para_mat, eyes_center)
        
        diff_coords = self.correct_pred_coords - eyes_pred_coords

        return diff_coords
    
    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def compute_diff(self, org_image, image,
                     face_start_point, face_end_point,
                     IsDrawing=True):
        gray = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(left=face_start_point[0],
                              top=face_start_point[1],
                              right=face_end_point[0],
                              bottom=face_end_point[1])
        # rect = [face_start_point, face_end_point]
        print(f'left:{face_start_point[0]}')
        print(f'top:{face_start_point[1]}')
        print(f'right:{face_end_point[0]}')
        print(f'bottom:{face_end_point[1]}')
        print(f'gray: {gray.shape} / {type(gray)} rect: {rect} / {type(rect)}')
    
        shape = self.landmark_predictor(gray, rect)
        shape = self.shape_to_np(shape)
        shape_for_eyes = np.array(shape[36:48])
        eyes_center = np.mean(shape_for_eyes, axis=0, dtype=np.int16)
        diff_coords = self.get_diff_from_eyes_center(eyes_center)
        text = f'diff_coords: {diff_coords}'

        min_correction = eyes_center[1] < self.correct_y_range[1]
        max_correction = eyes_center[1] > self.correct_y_range[0]
        IsEyesInArea = min_correction and max_correction

        if IsDrawing:
            for (x, y) in shape_for_eyes:
                # print(f'(x, y): ({x},{y})')
                cv2.circle(image, (x, y), 2, (0, 255, 0), 2)
            # print(f'eyes center: {eyes_center}')
            cv2.circle(image, eyes_center, 2, (255, 0, 0), 2)

            cv2.putText(
                image,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

            if IsEyesInArea:
                cv2.rectangle(image, (0, self.correct_y_range[0]),
                              (self.width, self.correct_y_range[1]),
                              (0, 255, 0), 3)

        return image