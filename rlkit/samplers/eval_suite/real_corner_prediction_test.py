from pandas.core import frame
from rlkit.samplers.eval_suite.base import EvalTest
import typing
import numpy as np
import pandas as pd
import os
from rlkit.samplers.eval_suite.utils import get_obs_preprocessor, create_real_corner_image_directories
import copy
from clothmanip.utils.utils import get_keys_and_dims
import cv2
from collections import deque


class RealCornerPredictionTest(EvalTest):
    def __init__(self, *args, variant, **kwargs):
        super().__init__(*args, variant=variant, **kwargs)
        self.save_images_every_epoch = variant['save_images_every_epoch']
        self.max_path_length = variant['algorithm_kwargs']['max_path_length']
        keys, _ = get_keys_and_dims(variant, self.env)
        self.obs_preprocessor = get_obs_preprocessor(keys['path_collector_observation_key'], variant['path_collector_kwargs']['additional_keys'], keys['desired_goal_key'])
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.camera_type = variant['env_kwargs']['camera_type']
        self.frame_stack_size = variant['env_kwargs']['frame_stack_size']


    def single_evaluation(self, eval_number: int) -> dict:
        print("Real corner eval", eval_number)
        create_real_corner_image_directories(self.base_save_folder, self.epoch)
        images_dir = os.path.join(self.file_dir, "images", self.camera_type)
        save_images = self.epoch % self.save_images_every_epoch == 0

        o = self.env.reset()
        corner_indices = self.env.corner_indices
        total_off = 0

        for i, image_dir in enumerate(os.listdir(images_dir)):
            image_dir_path = os.path.join(images_dir, image_dir)
            labels = pd.read_csv(f"{image_dir_path}/labels.csv", names=["corner", "u", "v", "file", "w", "h"])
            off_directory = 0
            frame_stack = deque([], maxlen = self.frame_stack_size)
            first_image_file_path = os.path.join(images_dir, image_dir, "1.png")
            first_image = cv2.imread(first_image_file_path)
            first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
            for _ in range(self.frame_stack_size):
                frame_stack.append(first_image.flatten()/255)

            for image_file in os.listdir(image_dir_path):
                image_index, suffix = image_file.split(".")
                if suffix == "png":
                    image_index = int(image_index)
                    image_file_path = os.path.join(images_dir, image_dir, image_file)
                    image = cv2.imread(image_file_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    w, h = image.shape
                    frame_stack.append(image.flatten()/255)
                    o['image'] = np.array([image for image in frame_stack]).flatten()
                    o_for_agent = self.obs_preprocessor(o)
                    a, agent_info, aux_output = self.policy.get_action(o_for_agent)

                    c0 = labels[(labels["corner"] == 0) & (labels["file"] == f'{image_index+1}.png')]
                    c1 = labels[(labels["corner"] == 1) & (labels["file"] == f'{image_index+1}.png')]
                    c2 = labels[(labels["corner"] == 2) & (labels["file"] == f'{image_index+1}.png')]
                    c3 = labels[(labels["corner"] == 3) & (labels["file"] == f'{image_index+1}.png')]

                    real_corners = [c0, c1, c2, c3]
                    sim_corners = aux_output.flatten()

                    for sim_corner_idx, sim_corner_idx_in_real in enumerate(corner_indices):
                        sim_corner = np.array([sim_corners[sim_corner_idx*2]*w, sim_corners[sim_corner_idx*2+1]*h]).flatten()
                        real_corner = np.array([real_corners[sim_corner_idx_in_real]['u'], real_corners[sim_corner_idx_in_real]['v']]).flatten()
                        if real_corner.shape[0] == 2:
                            off = np.linalg.norm(sim_corner-real_corner)
                        else:
                            off = 0
                        off_directory += off


                    if save_images and i == 0:
                        for aux_idx in range(int(aux_output.flatten().shape[0]/2)):
                            aux_u = int(aux_output.flatten()[aux_idx*2]*w)
                            aux_v = int(aux_output.flatten()[aux_idx*2+1]*h)
                            cv2.circle(image, (aux_u, aux_v), 2, (0, 255, 0), -1)
                        cv2.imwrite(f'{self.base_save_folder}/epochs/{self.epoch}/real_corners_prediction/{str(image_index).zfill(3)}.png', image)
            total_off += off_directory/image_index
        total_off /= i


        

            
        return dict(corner_error=total_off)        
