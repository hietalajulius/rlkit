from rlkit.samplers.eval_suite.base import EvalTest
import typing
import numpy as np
import pandas as pd
import os
from rlkit.samplers.eval_suite.utils import get_obs_preprocessor, create_blank_image_directories, save_blank_images
import copy
from clothmanip.utils.utils import get_keys_and_dims


class BlankImagesTest(EvalTest):
    def __init__(self, *args, variant, **kwargs):
        super().__init__(*args, variant=variant, **kwargs)
        self.save_images_every_epoch = variant['save_images_every_epoch']
        self.max_path_length = variant['algorithm_kwargs']['max_path_length']
        keys, _ = get_keys_and_dims(variant, self.env)
        self.obs_preprocessor = get_obs_preprocessor(keys['path_collector_observation_key'], variant['path_collector_kwargs']['additional_keys'], keys['desired_goal_key'])


    def single_evaluation(self, eval_number: int) -> dict:
        print("Blank eval", eval_number)
        save_images = (self.epoch % self.save_images_every_epoch == 0) and (eval_number == 0)

        if save_images:
            create_blank_image_directories(self.base_save_folder, self.epoch)

    
        path_length = 0
        o = self.env.reset()
        d = False
        while path_length < self.max_path_length:
            o['image'] = np.zeros(o['image'].shape)
            o_for_agent = self.obs_preprocessor(o)
            a, agent_info, aux_output = self.policy.get_action(o_for_agent)
            if save_images:
                save_blank_images(self.env, self.base_save_folder, self.epoch, path_length, aux_output)

            next_o, r, d, env_info = self.env.step(copy.deepcopy(a))
            path_length += 1

            if d:
                break
            o = next_o

        corner_distances = np.linalg.norm(next_o['achieved_goal']-next_o['desired_goal'])
        if d:
            return dict(success_rate=1.0, corner_distance=corner_distances)
        else:
            return dict(success_rate=0.0, corner_distance=corner_distances)