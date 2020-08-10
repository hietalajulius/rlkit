import numpy as np
from sideways_trajectory import items
import gym

def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
        get_action_kwargs=None,
        return_dict_obs=False,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if render:
        env.render(**render_kwargs)
    goal = o[desired_goal_key]
    while path_length < max_path_length:
        dict_obs.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        if render:
            env.render(**render_kwargs)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=dict_obs,
    ), np.any(np.array(rewards) == 0)

class Actor(object):
    def __init__(self):
        self.items = items
        self.current_action_idx = 0

    def get_action(self, obs, **kwargs):
        ac = np.array([0.0,0.0,0.0])
        if self.current_action_idx < len(self.items):
            action = self.items[self.current_action_idx]
            ac[0] = action[0] *0.85 + np.random.normal(0, 0.1)
            ac[1] = action[1] + np.random.normal(0, 0.1)
            ac[2] = action[2] * 1.5 + np.random.normal(0, 0.1)
            if self.current_action_idx > 32:
                ac[0] = 0
                ac[1] = 0
                ac[2] = -1
        self.current_action_idx += 1
        return ac, {}
    def reset(self):
        self.current_action_idx = 0


def main():
    env = gym.make('ClothSideways-v1')
    actor = Actor()
    num_examples = 100
    successes = 0
    try_n = 0
    rollouts = []
    while successes < num_examples:
        try_n += 1
        print("ITERATION NUMBER ", try_n, "Success so far", successes)
        rollout, success = multitask_rollout(env,actor,max_path_length=50, render=False,observation_key='observation',desired_goal_key='desired_goal',return_dict_obs=True)
        if success:
            successes += 1
            rollouts.append(rollout)



    fileName = "data_cloth_sideways_rlkit"
    fileName += "_" + str(num_examples)
    fileName += ".npz"

    np.savez_compressed(fileName, rollouts=rollouts) # save the file
    print("Saved, success rate:", successes)


if __name__ == "__main__":
    main()