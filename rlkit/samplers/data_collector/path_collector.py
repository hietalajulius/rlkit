from collections import deque, OrderedDict
from functools import partial

import numpy as np

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.base import PathCollector
from rlkit.samplers.rollout_functions import rollout
from rlkit.samplers.async_rollout_functions import vec_env_rollout



class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            use_demos=False,
            demo_path=None,
            num_demoers=0,
            demo_coef=1.0,
            rollout_fn=rollout,
            save_env_in_snapshot=False,  # WTFFF
    ):
        if render_kwargs is None:
            render_kwargs = {}

        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        self.use_demos=use_demos
        self.demo_path=demo_path
        self.num_demoers=num_demoers
        self.demo_coef=demo_coef

        self._num_steps_total = 0
        self._num_paths_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths
    ):
        paths = []
        num_steps_collected = 0
        demo_tries = 0
        demo_successes = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = max_path_length
            '''
            min(  
                max_path_length,
                num_steps - num_steps_collected,
            )
            '''

            path = self._rollout_fn(
                self._env,
                self._policy,
                use_demos=self.use_demos,
                demo_path=self.demo_path,
                demo_coef=self.demo_coef,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
            )
            path_len = len(path['actions'])
            if (
                    path_len != max_path_length
                    and not path['terminals'][-1]
                    and discard_incomplete_paths
            ):
                break
            
            if self.use_demos:
                demo_tries += 1
                successes = np.array([info['is_success']
                                      for info in path['env_infos']])
                if not np.any(successes):
                    print("Not successful demo", len(paths), demo_successes, "/", demo_tries)
                else:
                    demo_successes += 1
                    print("Demo success", len(paths), demo_successes, "/", demo_tries)
                    num_steps_collected += path_len
                    paths.append(path)
            else:
                num_steps_collected += path_len
                paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class KeyPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            additional_keys=[],
            env_timestep=None,
            new_action_every_ctrl_step=None,
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            obs = o[observation_key]
            for additional_key in additional_keys:
                obs = np.hstack((obs, o[additional_key]))

            return np.hstack((obs, o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self.env_timestep = env_timestep
        self.new_action_every_ctrl_step = new_action_every_ctrl_step
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot

class VectorizedKeyPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            output_max,
            observation_key='observation',
            desired_goal_key='desired_goal',
            additional_keys=[],
            goal_sampling_mode=None,
            processes=1,
            **kwargs
    ):
        def obs_processor(o):
            obs = o[observation_key]
            for key in additional_keys:
                obs = np.hstack((obs, o[key]))
            obs = np.hstack((obs, o[desired_goal_key]))
            return obs

        rollout_fn = partial(
            vec_env_rollout,
            output_max=output_max,
            processes=processes,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._additional_keys = additional_keys
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths=False,
    ):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        paths = []
        num_steps_collected = 0

        while num_steps_collected < num_steps:
            collected_paths = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length,
                use_demos=self.use_demos,
                demo_path=self.demo_path,
                num_demoers=self.num_demoers,
                demo_coef=self.demo_coef
            )
            collected_paths_len = 0
            for path in collected_paths:
                collected_paths_len += len(path['actions'])

            num_steps_collected += collected_paths_len
            paths += collected_paths

        # TODO: Above gives too many extra paths if early dones, which is ok ish
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        stripped_paths = []
        for path in paths:
            stripped_path = dict(
                rewards=path['rewards'], actions=path['actions'], env_infos=path['env_infos'])
            stripped_paths.append(stripped_path)

        self._epoch_paths.extend(stripped_paths)
        return paths

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env,
            policy,
            decode_goals=False,
            **kwargs
    ):
        """Expects env is VAEWrappedEnv"""
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)
