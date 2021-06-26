import numpy as np
import copy


def vec_env_rollout(
        env,
        agent,
        num_demoers=0,
        demo_paths=[],
        demo_divider=1,
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

    path_length = 0
    agent.reset()
    o = env.reset()

    done_indices = []
    use_demos = num_demoers > 0
    if use_demos:
        demo_path = np.random.choice(demo_paths)
        predefined_actions = np.genfromtxt(demo_path, delimiter=',')
        

    while path_length < max_path_length:
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, _ = agent.get_actions(o_for_agent, **get_action_kwargs)
        agent_info = [{} for _ in range(processes)]

        if use_demos:
            if path_length < predefined_actions.shape[0]:
                delta = np.random.normal(predefined_actions[path_length][:3], 0.01)
            else:
                delta = np.zeros(3)

            delta = np.clip(delta/demo_divider, -1, 1)
            a[:num_demoers] = delta

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        for idx, path_dict in enumerate(paths):
            if not idx in done_indices:
                if d[idx]:
                    done_indices.append(idx)

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

        if len(done_indices) == processes:
            break

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
