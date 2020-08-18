import numpy as np
import gym
from rlkit.samplers.rollout_functions import multitask_rollout


class Actor(object):
    def __init__(self):
        self.current_action_idx = 0
    def get_action(self, obs, **kwargs):
        action = np.array([0.0, 0.0, 0.0])
        if self.current_action_idx < 12:
            action[0] = 0.2
            action[1] = 0.2
            action[2] = 0.45
        elif self.current_action_idx < 40:
            action[0] = 0.2
            action[1] = 0.2
            action[2] = -0.2
        else:
            action[2] = -1
        self.current_action_idx += 1
        action += np.random.normal(0, 0.1, 3)

        #action = np.array([-1.,1.,0.])
        return action, np.random.normal(0, 0.1, 12), {}
    def reset(self):
        self.current_action_idx = 0



def make_demo_rollouts(env_name, num_examples,):
    env = gym.make(env_name)
    actor = Actor()
    
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

    env_name = 'ClothDiagonal-v1'
    num_examples = 100
    rollouts = make_demo_rollouts(env_name,num_examples)
    file_name = "data_cloth_diagonal_rlkit"
    file_name += "_" + str(num_examples)
    file_name += "_" + env_name
    file_name += ".npz"

    np.savez_compressed(file_name, rollouts=rollouts) # save the file
    print("Saved")