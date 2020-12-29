import numpy as np
import copy


def vec_env_rollout(
        env,
        agent,
        processes=1,
        max_path_length=np.inf,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs={}
):
    if preprocess_obs_for_policy_fn is None:
        def preprocess_obs_for_policy_fn(x):
            return x
    paths = []
    for _ in range(processes):
        path = dict(
            observations=[],
            actions=[],
            rewards=[],
            next_observations=[],
            terminals=[],
            agent_infos=[],
            env_infos=[],
        )
        paths.append(path)

    #raw_obs = []
    #raw_next_obs = []
    path_length = 0
    agent.reset()
    o = env.reset()
    # TODO: enough to reset?

    while path_length < max_path_length:
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a = agent.get_actions(o_for_agent, **get_action_kwargs)
        agent_info = [{} for _ in range(processes)]

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        #print("Async step", d.shape)

        for idx, path_dict in enumerate(paths):
            obs_dict = dict()
            next_obs_dict = dict()
            for key in o.keys():
                obs_dict[key] = o[key][idx]
                next_obs_dict[key] = next_o[key][idx]
            path_dict['observations'].append(obs_dict)
            path_dict['rewards'].append(r[idx])
            path_dict['terminals'].append(d[idx])
            path_dict['actions'].append(a[idx])
            path_dict['next_observations'].append(next_obs_dict)
            path_dict['agent_infos'].append(agent_info[idx])
            path_dict['env_infos'].append(env_info[idx])

        path_length += 1
        # if d:
        # break
        # TODO Figure out terminals handling
        o = next_o

    for idx, path_dict in enumerate(paths):
        path_dict['actions'] = np.array(path_dict['actions'])
        path_dict['observations'] = np.array(path_dict['observations'])
        path_dict['next_observations'] = np.array(
            path_dict['next_observations'])
        path_dict['rewards'] = np.array(path_dict['rewards'])
        path_dict['terminals'] = np.array(
            path_dict['terminals']).reshape(-1, 1)

        if len(path_dict['actions'].shape) == 1:
            path_dict['actions'] = np.expand_dims(path_dict['actions'], 1)

        if len(path_dict['rewards'].shape) == 1:
            path_dict['rewards'] = path_dict['rewards'].reshape(-1, 1)

    return paths
