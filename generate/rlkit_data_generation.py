import numpy as np
import gym
from rlkit.samplers.rollout_functions import multitask_rollout
from generate.sideways_trajectory import items2
from generate.actor import Actor
from pynput import mouse
import copy

class SidewaysActor(object):
    def __init__(self):
        self.items = items2
        self.current_action_idx = 0

    def get_action(self, obs, **kwargs):
        ac = np.array([0.0,0.0,0.0])
        if self.current_action_idx < len(self.items):
            action = copy.deepcopy(self.items[self.current_action_idx])
            action[0] *= 0.75
            action[1] *= 0.8
            action[2] *= 4
            ac = action 
        else:
            ac[0] = -0.05
            ac[2] = -0.5

        self.current_action_idx += 1
        ac += np.random.normal(0, 0.1, 3)
        return ac, np.random.normal(0, 0.1, 12), {}
    def reset(self):
        self.current_action_idx = 0


class DiagonalActor(object):
    def __init__(self):
        self.current_action_idx = 0
    def get_action(self, obs, **kwargs):
        action = np.array([0.0, 0.0, 0.0])
        if self.current_action_idx < 12:
            action[0] = -0.3
            action[1] = 0.3
            action[2] = 0.45
        elif self.current_action_idx < 40:
            action[0] = -0.2
            action[1] = 0.2
            action[2] = -0.2
        else:
            action[2] = -1
        self.current_action_idx += 1
        action += np.random.normal(0, 0.1, 3)
        return action, np.random.normal(0, 0.1, 12), {}
    def reset(self):
        self.current_action_idx = 0



def make_demo_rollouts(env_name, num_examples, env_type=None):
    env = gym.make(env_name)
    if env_type == 'sideways':
        actor = SidewaysActor()
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
        print("ITERATION NUMBER ", try_n, "Success so far", successes)
        rollout = multitask_rollout(env,actor,max_path_length=50,observation_key='observation',desired_goal_key='desired_goal',return_dict_obs=True)
        success = np.any(np.array(rollout['rewards']) == 0)
        if success:
            successes += 1
            rollouts.append(rollout)

    return rollouts
    


if __name__ == "__main__":

    env_name = 'ClothSidewaysStrict-v1'
    num_examples = 100
    rollouts = make_demo_rollouts(env_name,num_examples, env_type='sideways')
    file_name = "data_cloth_diagonal_rlkit"
    file_name += "_" + str(num_examples)
    file_name += "_" + env_name
    file_name += ".npz"

    np.savez_compressed(file_name, rollouts=rollouts) # save the file
    print("Saved")