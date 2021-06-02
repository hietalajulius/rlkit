from functools import partial

import numpy as np
import copy
import os
import glob
import cv2

create_rollout_function = partial


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
        full_o_postprocess_func=None,
):
    if full_o_postprocess_func:
        def wrapped_fun(env, agent, o):
            full_o_postprocess_func(env, agent, observation_key, o)
    else:
        wrapped_fun = None

    def obs_processor(o):
        return np.hstack((o[observation_key], o[desired_goal_key]))

    paths = rollout(
        env,
        agent,
        max_path_length=max_path_length,
        render=render,
        render_kwargs=render_kwargs,
        get_action_kwargs=get_action_kwargs,
        preprocess_obs_for_policy_fn=obs_processor,
        full_o_postprocess_func=wrapped_fun,
    )
    if not return_dict_obs:
        paths['observations'] = paths['observations'][observation_key]
    return paths


def contextual_rollout(
        env,
        agent,
        observation_key=None,
        context_keys_for_policy=None,
        obs_processor=None,
        **kwargs
):
    if context_keys_for_policy is None:
        context_keys_for_policy = ['context']

    if not obs_processor:
        def obs_processor(o):
            combined_obs = [o[observation_key]]
            for k in context_keys_for_policy:
                combined_obs.append(o[k])
            return np.concatenate(combined_obs, axis=0)
    paths = rollout(
        env,
        agent,
        preprocess_obs_for_policy_fn=obs_processor,
        **kwargs
    )
    return paths

def rollout(
        env,
        agent,
        demo_coef=1.0,
        use_demos=False,
        demo_path=None,
        max_path_length=np.inf,
        save_folder=None,
        env_timestep=None,
        new_action_every_ctrl_step=None,
        evaluate=False,
        epoch=0,
        render=False,
        save_images_every_epoch=1,
        render_kwargs=None,
        preprocess_obs_for_policy_fn=None,
        get_action_kwargs=None,
        return_dict_obs=False,
        full_o_postprocess_func=None,
        reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        def preprocess_obs_for_policy_fn(x): return x
    raw_obs = []
    raw_next_obs = []
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


    if use_demos:
        predefined_actions = np.genfromtxt(demo_path, delimiter=',')*demo_coef

    if evaluate and epoch % save_images_every_epoch == 0:
        try:
            cnn_path = f"{save_folder}/images/{epoch}/cnn"
            corners_path = f"{save_folder}/images/{epoch}/corners"
            eval_path = f"{save_folder}/images/{epoch}/eval"
            os.makedirs(cnn_path)
            os.makedirs(corners_path)
            os.makedirs(eval_path)
        except:
            print("folders existed already")
        trajectory_log = []
        trajectory_log.append(np.concatenate([env.desired_pos_step_W, env.desired_pos_ctrl_W, env.get_ee_position_I(), env.get_ee_position_W(), np.zeros(9)]))


    if reset_callback:
        reset_callback(env, agent, o)

    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info, aux_output = agent.get_action(
            o_for_agent, **get_action_kwargs)

        if use_demos:
            if path_length < predefined_actions.shape[0]:
                delta = np.random.normal(predefined_actions[path_length][:3], 0.01)
            else:
                delta = np.zeros(3)
            a = delta/env.output_max
            a = np.clip(a, -1, 1)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        if evaluate and epoch % save_images_every_epoch == 0:
            train_image, eval_image = env.capture_image(aux_output)
            cv2.imwrite(f'{save_folder}/images/{epoch}/corners/{str(path_length).zfill(3)}.png', train_image)
            cv2.imwrite(f'{save_folder}/images/{epoch}/eval/{str(path_length).zfill(3)}.png', eval_image)

            if "image" in o.keys():
                data = o['image'].copy().reshape((-1, 100, 100))
                for i, image in enumerate(data):
                    reshaped_image = image.reshape(100,100, 1)
                    cv2.imwrite(f'{save_folder}/images/{epoch}/cnn/{str(path_length).zfill(3)}_{i}.png', reshaped_image*255)


        next_o, r, d, env_info = env.step(copy.deepcopy(a))

        if evaluate:
            delta = env.get_ee_position_W() - trajectory_log[-1][9:12]
            
            velocity = delta / (env_timestep*new_action_every_ctrl_step)
            acceleration = (velocity - trajectory_log[-1][15:18]) / (env_timestep*new_action_every_ctrl_step)
            
            trajectory_log.append(np.concatenate([env.desired_pos_step_W, env.desired_pos_ctrl_W, env.get_ee_position_I(), env.get_ee_position_W(), delta, velocity, acceleration]))

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o

    if evaluate:
        np.savetxt(f"{save_folder}/eval_trajs/{epoch}.csv",
                    trajectory_log, delimiter=",", fmt='%f')
        np.savetxt(f"{save_folder}/eval_trajs/executable_deltas_{epoch}.csv",
                    np.array(trajectory_log)[:,12:15], delimiter=",", fmt='%f')

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        full_observations=raw_obs,
        full_next_observations=raw_obs,
    )


def deprecated_rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
