import gym
import copy
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.ddpg.ddpg import DDPGTrainer
from rlkit.torch.her.her import HERTrainer
from rlkit.samplers.data_collector import GoalConditionedPathCollector
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpislonStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer


def experiment(variant):
    eval_env = gym.make('ClothSideways-v1').env

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size

    replay_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    '''
    demo_buffer = ObsDictRelabelingBuffer(
        env=eval_env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    demo_buffer.add_paths_from_file()
    '''

    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
    )

    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    paths = eval_path_collector.collect_new_paths(
            50,
            100,
            discard_incomplete_paths=False,
            render=False
        )
    print("Paths keys", paths[0].keys())
    for key in paths[0].keys():
        print(key, len(paths[0][key]))
    #print(paths[0]['full_observations'])
    #for path in paths:
        #replay_buffer.add_path(path)

    print("Obbs pix", paths[0]['observations'][0].keys())
    print("Buf buf", replay_buffer._size)




if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm='HER-DDPG',
        version='normal',
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=10,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=50,
            num_trains_per_train_loop=40,
            num_train_loops_per_epoch=50,
            min_num_steps_before_training=500,
            max_path_length=50,
        ),
        ddpg_trainer_kwargs=dict(
            use_soft_update=True,
            tau=5e-1,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-3,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256],
        ),
    )
    experiment(variant)