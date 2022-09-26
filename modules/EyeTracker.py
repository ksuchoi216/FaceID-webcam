import numpy as np
import json


class EyeTracker(object):
    def __init__(self, cfg):
        # self.single_face_detector = single_face_detector
        
        folder_for_preset_images = cfg['folder_for_preset_images']
        self.correct_area = np.asarray(cfg['correct_area'])
        print(f'correct area: \n{self.correct_area}')

        first_move_setting = cfg['first_move_setting']
        second_move_setting = cfg['second_move_setting']

        first_move_setting = np.asarray(first_move_setting)
        second_move_setting = np.asarray(second_move_setting)

        move_mat = np.stack((first_move_setting, second_move_setting), axis=-1)

        #TODO: to add face center functions(get face centers)
        face_center1 = np.array([1, 0])
        face_center2 = np.array([0, 2])
        face_center3 = np.array([-2, 1])
        face_centers = [face_center1, face_center2, face_center3]

        delta_vec1 = face_centers[1] - face_centers[0]
        delta_vec2 = face_centers[2] - face_centers[1]

        delta_mat = np.stack((delta_vec1, delta_vec2), axis=-1)

        delta_mat_inv = np.linalg.inv(delta_mat)
        # print(np.allclose(np.dot(delta_mat, delta_mat_inv), np.eye(2)))
        # print(np.allclose(np.dot(delta_mat_inv, delta_mat), np.eye(2)))

        self.para_mat = np.matmul(move_mat, delta_mat_inv)
        
        print(f'parameters matrix was calculated: \n {self.para_mat}')

    def predict_coordinates(self, coordinates):
        # coordinates e.g. = [400, 200]
        pred_coords = np.matmul(self.para_mat, coordinates)
        
        return pred_coords
    
    def calculate_differences(self, face_center):
        face_pred_coords = self.predict_coordinates(face_center)
        
        correct_coords = np.mean(self.correct_area, axis=0, dtype=np.int16)
        correct_pred_coords = self.predict_coordinates(correct_coords)
        
        diff_coords = correct_pred_coords - face_pred_coords
        
        return diff_coords
    
    
if __name__ == "__main__":
    path_config = "./configs/config.json"
    with open(path_config) as f:
        cfg = json.load(f)

    eyeTracker = EyeTracker(cfg['EyeTracker'])

    face_center = np.array([2, 3])
    diff_coords = eyeTracker.calculate_differences(face_center)
    print(diff_coords)
