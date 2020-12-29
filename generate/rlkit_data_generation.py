# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#
import numpy as np
import gym
from gym.envs.robotics import task_definitions
from rlkit.samplers.rollout_functions import multitask_rollout
from generate.sideways_trajectory import items2
from pynput import mouse
import copy
import math


class SidewaysActor(object):
    def __init__(self):
        self.items = items2
        self.current_action_idx = 0

    def get_action(self, obs, **kwargs):
        ac = np.array([0.0, 0.0, 0.0])
        if self.current_action_idx < len(self.items):
            action = copy.deepcopy(self.items[self.current_action_idx])
            # action.append(1)
            action[0] *= 0.75
            action[1] *= 0.8
            action[2] *= 4
            ac = action
        else:
            ac[0] = -0.05
            ac[2] = -0.5
        self.current_action_idx += 1
        return ac, {}

    def reset(self):
        self.current_action_idx = 0


class DiagonalActor(object):
    def __init__(self):
        self.current_action_idx = 0

    def get_action(self, obs, **kwargs):
        action = np.array([0.0, 0.0, 0.0])
        if self.current_action_idx < 10:
            action[0] = -0.1
            action[1] = 0.1
            action[2] = 0.2

        elif self.current_action_idx < 35:
            action[0] = -0.2
            action[1] = 0.2
            action[2] = -0.2
        else:
            pass

        self.current_action_idx += 1
        return action, {}

    def reset(self):
        self.current_action_idx = 0


class SlipActor(object):
    def __init__(self):
        self.current_action_idx = 0

    def get_action(self, obs, **kwargs):
        action = np.array([0.0, 0.0, 0.0])
        if self.current_action_idx < 6:
            action[2] = 0.2
        elif self.current_action_idx < 22:
            action[0] = -0.1
            action[1] = 0.1
        else:
            action[0] = -0.1
            action[1] = -0.1

        self.current_action_idx += 1
        return action, {}

    def reset(self):
        self.current_action_idx = 0


def make_demo_rollouts(env_name, num_examples, env_type=None, render=False, **env_kwargs):
    env = gym.make(env_name, **env_kwargs)
    if env_type == 'sideways':
        actor = SidewaysActor()
    elif env_type == 'slip':
        actor = SlipActor()
    else:
        actor = DiagonalActor()
    '''
    actor = Actor(env)
    listener = mouse.Listener(
        on_move=actor.on_move,
        on_click=actor.on_click,
        on_scroll=actor.on_scroll)
    listener.start()
    env.set_key_callback_function(actor.set_values)
    '''

    successes = 0
    try_n = 0
    rollouts = []
    while successes < num_examples:
        try_n += 1

        env.set_geom_params()
        print("set geom params")
        print("ITERATION NUMBER ", try_n, "Success so far", successes)
        rollout = multitask_rollout(env, actor, render=render, max_path_length=100,
                                    observation_key='observation', desired_goal_key='desired_goal', return_dict_obs=True)

        success = np.any(np.array(rollout['rewards']) == 0)
        if success:
            successes += 1
            rollouts.append(rollout)

    return rollouts


if __name__ == "__main__":

    env_name = 'Cloth-v1'
    env_kwargs = dict(
        constraints=task_definitions.constraints['diagonal_1'],
        sparse_dense=False,
        pixels=False,
        image_size=500,
        randomize_params=False,
        randomize_geoms=False,
        max_advance=0.05,
        random_seed=1

    )
    num_examples = 100
    rollouts = make_demo_rollouts(
        env_name, num_examples, "slip", render=True, **env_kwargs)
    file_name = "rlkit_demo_data"
    file_name += "_" + str(num_examples)
    file_name += "_" + env_name
    file_name += ".npz"

    np.savez_compressed(file_name, rollouts=rollouts)  # save the file
    print("Saved")
