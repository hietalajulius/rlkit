from rlkit.samplers.eval_suite.base import EvalTest
import typing
import numpy as np
import pandas as pd
import os
from rlkit.samplers.eval_suite.utils import get_obs_preprocessor, create_regular_image_directories, save_regular_images
import copy
from clothmanip.utils.utils import get_keys_and_dims
import cv2


class SuccessRateTest(EvalTest):
    def __init__(self, *args, variant, **kwargs):
        super().__init__(*args, variant=variant, **kwargs)
        self.save_images_every_epoch = variant['save_images_every_epoch']
        self.max_path_length = variant['algorithm_kwargs']['max_path_length']
        keys, _ = get_keys_and_dims(variant, self.env)
        self.obs_preprocessor = get_obs_preprocessor(keys['path_collector_observation_key'], variant['path_collector_kwargs']['additional_keys'], keys['desired_goal_key'])
        self.save_blurred_images = variant['env_kwargs']['randomization_kwargs']['albumentations_randomization']


    def single_evaluation(self, eval_number: int) -> dict:
        print(f"{self.name} eval", eval_number)
        trajectory_log = pd.DataFrame()
        save_images = (self.epoch % self.save_images_every_epoch == 0) and (eval_number == 0)

        if save_images:
            create_regular_image_directories(self.base_save_folder, self.name, self.epoch)
            blurred_path = f"{self.base_save_folder}/epochs/{self.epoch}/{self.name}/blurred_cnn"
            try:
                os.makedirs(blurred_path)
            except:
                print("blurred folders existed already")

    
        path_length = 0

        self.env.randomization_kwargs['cloth_type'] = self.name
        self.env.randomize_xml_model()
        
        o = self.env.reset()
        d = False
        success = False
        while path_length < self.max_path_length:
            o_for_agent = self.obs_preprocessor(o)
            a, agent_info, aux_output = self.policy.get_action(o_for_agent)

            image = o['image']
            if self.save_blurred_images and save_images:
                image = image.reshape((-1, 100, 100))*255
                cv2.imwrite(f"{self.base_save_folder}/epochs/{self.epoch}/{self.name}/blurred_cnn/{str(path_length).zfill(3)}.png", image[0])


            if save_images:
                save_regular_images(self.env, self.base_save_folder, self.name, self.epoch, path_length, aux_output)
            trajectory_log = trajectory_log.append(
                    self.env.get_trajectory_log_entry(), ignore_index=True)

            next_o, r, d, env_info = self.env.step(copy.deepcopy(a))
            path_length += 1
            if env_info['is_success']:
                success = True

            if d:
                break
            o = next_o

        trajectory_log.to_csv(f"{self.base_save_folder}/epochs/{self.epoch}/trajectory.csv")
        trajectory_log['raw_action'].to_csv(f"{self.base_save_folder}/epochs/{self.epoch}/executable_raw_actions.csv")
        corner_distances = np.linalg.norm(next_o['achieved_goal']-next_o['desired_goal'])
        if success:
            return dict(success_rate=1.0, corner_distance=corner_distances)
        else:
            return dict(success_rate=0.0, corner_distance=corner_distances)
            
        
        




    
