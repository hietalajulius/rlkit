from pandas.core import frame
from rlkit.samplers.eval_suite.base import EvalTest
import typing
import numpy as np
import pandas as pd
import os
from rlkit.samplers.eval_suite.utils import get_obs_preprocessor, create_real_corner_image_dump_directories
import copy
from clothmanip.utils.utils import get_keys_and_dims
import cv2
from collections import deque


class DumpRealCornerPredictions(EvalTest):
    def __init__(self, *args, variant, **kwargs):
        super().__init__(*args, variant=variant, **kwargs)
        self.save_images_every_epoch = variant['save_images_every_epoch']
        keys, _ = get_keys_and_dims(variant, self.env)
        self.obs_preprocessor = get_obs_preprocessor(keys['path_collector_observation_key'], variant['path_collector_kwargs']['additional_keys'], keys['desired_goal_key'])
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.frame_stack_size = variant['env_kwargs']['frame_stack_size']


    def single_evaluation(self, eval_number: int) -> dict:
        save_images = self.epoch % self.save_images_every_epoch == 0
        if save_images:
            print("Real corner dump", eval_number)

            o = self.env.reset()

            folder_dir = os.path.join(self.file_dir, "images", "livingroom")
            for folder in os.listdir(folder_dir):
                create_real_corner_image_dump_directories(self.base_save_folder, folder, self.epoch)
                frame_stack = deque([], maxlen = self.frame_stack_size)
                first_image_file_path = os.path.join(folder_dir, folder, "0.png")
                first_image = cv2.imread(first_image_file_path)
                first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
                for _ in range(self.frame_stack_size):
                        frame_stack.append(first_image.flatten()/255)

                for file in sorted(os.listdir(os.path.join(folder_dir, folder))):
                    image_index, suffix = file.split(".")
                    if suffix == "png":
                        image_index = int(image_index)
                        image_file_path = os.path.join(folder_dir, folder, file)
                        image = cv2.imread(image_file_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        w, h = image.shape
                        frame_stack.append(image.flatten()/255)
                        o['image'] = np.array([image for image in frame_stack]).flatten()
                        o_for_agent = self.obs_preprocessor(o)
                        a, agent_info, aux_output = self.policy.get_action(o_for_agent)
                        for aux_idx in range(int(aux_output.flatten().shape[0]/2)):
                            aux_u = int(aux_output.flatten()[aux_idx*2]*w)
                            aux_v = int(aux_output.flatten()[aux_idx*2+1]*h)
                            cv2.circle(image, (aux_u, aux_v), 2, (0, 255, 0), -1)
                        cv2.imwrite(f'{self.base_save_folder}/epochs/{self.epoch}/real_corners_dump/{folder}/{str(image_index).zfill(3)}.png', image)


        

            
        return {}      
