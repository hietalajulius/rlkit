import numpy as np
from gym.spaces import Dict, Discrete

from rlkit.data_management.replay_buffer import ReplayBuffer
import copy
from clothmanip.utils import task_definitions, reward_calculation


class FutureObsDictRelabelingBuffer(ReplayBuffer):
    """
    Replay buffer for environments whose observations are dictionaries, such as
        - OpenAI Gym GoalEnv environments.
          https://blog.openai.com/ingredients-for-robotics-research/
        - multiworld MultitaskEnv. https://github.com/vitchyr/multiworld/

    Implementation details:
     - Only add_path is implemented.
     - Image observations are presumed to start with the 'image_' prefix
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            ob_spaces,
            action_space,
            fraction_goals_rollout_goals=1.0,
            internal_keys=None,
            goal_keys=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
    ):
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        if goal_keys is None:
            goal_keys = []
        if desired_goal_key not in goal_keys:
            goal_keys.append(desired_goal_key)
        self.goal_keys = goal_keys

        assert fraction_goals_rollout_goals >= 0
        self.max_size = max_size
        self.task_reward_function = self.set_task_reward_function(None)
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals

        self.ob_keys_to_save = [
            observation_key,
            desired_goal_key,
            achieved_goal_key,
        ]
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        self.action_space = action_space
        self.ob_spaces = ob_spaces

        if isinstance(self.action_space, Discrete):
            self._action_dim = action_space.n
        else:
            self._action_dim = action_space.low.size

        self._actions = np.zeros((max_size, self._action_dim))
        self._control_penalties = np.zeros((max_size, 1))
        self._corner_positions = np.zeros((max_size, 8))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_size, 1), dtype='uint8')
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}

        for key in self.ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, "Key not found in the observation space: %s" % key
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.zeros(
                (max_size, self.ob_spaces[key].low.size), dtype=type)

        self._top = 0
        self._size = 0

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def set_task_reward_function(self, task_reward_function):
        self.task_reward_function = task_reward_function

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        control_penalties = np.array([info['control_penalty']
                                      for info in path['env_infos']])
        corner_positions = np.array([info['corner_positions']
                                      for info in path['env_infos']])



        path_len = len(terminals)

        actions = flatten_n(actions)
        corner_positions = flatten_n(corner_positions)
        control_penalties = flatten_n(control_penalties)
        if isinstance(self.action_space, Discrete):
            actions = np.eye(self._action_dim)[actions]
            actions = actions.reshape((-1, self._action_dim))
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(
            next_obs,
            self.ob_keys_to_save + self.internal_keys,
        )
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        if self._top + path_len >= self.max_size:
            """
            All of this logic is to handle wrapping the pointer when the
            replay buffer gets full.
            """
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                self._top:self._top + num_pre_wrap_steps, :
            ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                self._corner_positions[buffer_slice] = corner_positions[path_slice]
                self._control_penalties[buffer_slice] = control_penalties[path_slice]
                
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = (
                        next_obs[key][path_slice]
                    )
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self.max_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            self._control_penalties[slc] = control_penalties
            self._corner_positions[slc] = corner_positions
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - num_rollout_goals

        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_future_goals > 0:
            future_indices = indices[-num_future_goals:]
            possible_future_obs_lens = np.array([
                len(self._idx_to_future_obs_idx[i]) for i in future_indices
            ])
            # Faster than a naive for-loop.
            # See https://github.com/vitchyr/rlkit/pull/112 for details.
            next_obs_idxs = (
                np.random.random(num_future_goals) * possible_future_obs_lens
            ).astype(np.int)
            future_obs_idxs = np.array([
                self._idx_to_future_obs_idx[ids][next_obs_idxs[i]]
                for i, ids in enumerate(future_indices)
            ])


            resampled_goals[-num_future_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]
        corner_positions = self._corner_positions[indices]
        new_task_rewards = self.task_reward_function(
            new_next_obs_dict[self.achieved_goal_key],
            new_next_obs_dict[self.desired_goal_key],
            dict(num_future_goals=num_future_goals)
        )

        control_penalties = self._control_penalties[indices].flatten()
        new_rewards = new_task_rewards + control_penalties

        new_rewards = new_rewards.reshape(-1, 1)
        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]

        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
            'corner_positions': corner_positions
        }
        if 'image' in self.internal_keys:
            batch['images'] = new_obs_dict['image']
            batch['next_images'] = new_next_obs_dict['image']
        if 'robot_observation' in self.internal_keys:
            batch['robot_observations'] = new_obs_dict['robot_observation']
            batch['next_robot_observations'] = new_next_obs_dict['robot_observation']
        if 'model_params' in self.internal_keys:
            batch['model_params'] = new_obs_dict['model_params']

        return batch

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save + self.internal_keys
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save + self.internal_keys
        }


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }


def preprocess_obs_dict(obs_dict):
    """
    Apply internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unnormalize_image(obs)
    return obs_dict


def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict


def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
