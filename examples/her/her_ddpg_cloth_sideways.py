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
    expl_env = gym.make('ClothSideways-v1').env

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'


    achieved_goal_key = desired_goal_key.replace("desired", "achieved")
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
        **variant['demo_buffer_kwargs']
    )
    

    demo_buffer.add_paths_from_file()
    '''
    obs_dim = eval_env.observation_space.spaces['observation'].low.size
    action_dim = eval_env.action_space.low.size
    goal_dim = eval_env.observation_space.spaces['desired_goal'].low.size
    es = GaussianAndEpislonStrategy(
        action_space=expl_env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.1,
    )
    qf = FlattenMlp(
        input_size=obs_dim + action_dim + goal_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    normalizer = TorchFixedNormalizer(obs_dim + goal_dim, eps=0.01, default_clip_range=5)
    policy = TanhMlpPolicy(
        input_size=obs_dim + goal_dim,
        obs_normalizer=normalizer,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    target_qf = copy.deepcopy(qf)
    target_policy = copy.deepcopy(policy)
    trainer = DDPGTrainer(
        qf=qf,
        target_qf=target_qf,
        policy=policy,
        target_policy=target_policy,
        **variant['ddpg_trainer_kwargs']
    )
    trainer = HERTrainer(trainer)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    expl_path_collector = GoalConditionedPathCollector(
        expl_env,
        expl_policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    eval_path_collector = GoalConditionedPathCollector(
        eval_env,
        policy,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
    )
    '''
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        demo_buffer=demo_buffer,
        **variant['algo_kwargs']
    )
    '''
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

    eval_path_collector.collect_new_paths(
            50,
            1000,
            discard_incomplete_paths=True,
            render=True
        )



if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm='HER-DDPG',
        version='normal',
        algo_kwargs=dict(
            batch_size=1024,
            num_epochs=200,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=50,
            num_trains_per_train_loop=40,
            num_train_loops_per_epoch=20,
            min_num_steps_before_training=0,
            max_path_length=50,
        ),
        ddpg_trainer_kwargs=dict(
            use_soft_update=True,
            target_soft_update_period=40,
            tau=0.05,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-3,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0,
        ),
        demo_buffer_kwargs=dict(
            max_size=int(100),
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256],
        ),
    )
    setup_logger('her-ddpg-demos', variant=variant)
    experiment(variant)