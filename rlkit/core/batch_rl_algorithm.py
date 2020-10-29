import abc
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import time
import torch
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.samplers.rollout_functions import rollout
import gym
import numpy as np
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from multiprocessing import Process, Queue, Manager
from multiprocessing.managers import BaseManager
import threading
from rlkit.torch.sac.policies import TanhGaussianPolicy


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            demo_buffer: ReplayBuffer = None,
            demo_paths = None,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            demo_buffer,
            demo_paths
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):  
            print("Saving current model")
            torch.save(self.trainer._base_trainer.policy.state_dict(),'current_policy.mdl')
            print("Saved current model")
            print("Evaluation sampling")
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True
            )
            print("Evaluation done")
            gt.stamp('evaluation sampling')
            print("Epoch", epoch)
            collect_times = 0
            train_times = 0
            total_times = 0
            for cycle in range(self.num_train_loops_per_epoch):
                print("\n Cycle", cycle, epoch)
                start_collect = time.time()
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                col_time = time.time() - start_collect
                print("Took to collect:", col_time)
                collect_times += col_time
                gt.stamp('exploration sampling', unique=False)
                #for path in new_expl_paths:
                    #print("Added episode", len(path['observations']))

                self.replay_buffer.add_paths(new_expl_paths)
                #print("Replay buf", self.replay_buffer._size)
                gt.stamp('data storing', unique=False)

                start_train = time.time()
                self.training_mode(True)
                for tren in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    if not self.demo_buffer == None:
                        demo_data = self.demo_buffer.random_batch(int(self.batch_size*(1/8)))
                        self.trainer.train(train_data, demo_data)
                    else:
                        self.trainer.train(train_data)
                #print("Trained for", tren + 1, "times")
                tra_time = time.time() - start_train
                print("Took to train:", tra_time)
                train_times += tra_time
                gt.stamp('training', unique=False)
                self.training_mode(False)
                total_times += time.time() - start_collect
            print("Time collect avg cycle:", collect_times/self.num_train_loops_per_epoch)
            print("Time train avg cycle:", train_times/self.num_train_loops_per_epoch)
            print("Total avg cycle:", total_times/self.num_train_loops_per_epoch)
            print("Ending epoch")
            self._end_epoch(epoch)



def get_paths(new_path_queue, buffer):
    while True:
        new_path = new_path_queue.get()  
        #print("Read a new path")
        buffer.add_path(new_path)


def add_batch(batch_queue, buffer):
    while True:
        if buffer._size > 0:
            batch = buffer.random_batch(1000)
            #print("Put batch to queue")
            batch_queue.put(batch)


def path_collector_process(path_queue, namespace, index):
    kwargs = dict(
            task="sideways",
            pixels=False,
            strict=True,
            distance_threshold=0.05,
            randomize_params=False,
            randomize_geoms=False,
            uniform_jnt_tend=True,
            max_advance=0.05,
            random_seed=1
        )
    env = NormalizedBoxEnv(gym.make('Cloth-v1', **kwargs))

    def obs_processor(o):
        obs = o['observation']
        obs = np.hstack((obs, o['model_params'], o['desired_goal']))
        return obs

    while True:
        current_policy = namespace.policy
        print("Policy version", current_policy.fuba)
        path = rollout(env, current_policy, max_path_length=50, preprocess_obs_for_policy_fn=obs_processor)
        path_queue.put(path)



def buffer_process(path_queue, batch_queue):
    """
    A consumer that pops results off the resultqueue and prints them to screen
    """
    while True:
        pass

class AsyncBatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            demo_buffer: ReplayBuffer = None,
            demo_paths = None,
            num_processes = 3
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            demo_buffer,
            demo_paths
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_processes = num_processes

    def _train(self):

        with Manager() as manager:
            collection_ns = manager.Namespace()

            collection_ns.policy = self.trainer._base_trainer.policy
            
            path_queue = Queue()
            #batch_queue = Queue()

            collectors = [Process(target=path_collector_process, args=(path_queue, collection_ns, i)) for i in range(self.num_processes)]
            for c in collectors:
                c.start()
                
            kwargs = dict(
                task="sideways",
                pixels=False,
                strict=True,
                distance_threshold=0.05,
                randomize_params=False,
                randomize_geoms=False,
                uniform_jnt_tend=True,
                max_advance=0.05,
                random_seed=1
            )

            buffer_env = NormalizedBoxEnv(gym.make('Cloth-v1', **kwargs))
            buffer_kwargs = dict(
                max_size=100000,
                fraction_goals_env_goals = 0,
                fraction_goals_rollout_goals = 0.2,
                internal_keys = ['model_params']
            )

            buffer = ObsDictRelabelingBuffer(
                    env=buffer_env,
                    observation_key='observation',
                    desired_goal_key='desired_goal',
                    achieved_goal_key='achieved_goal',
                    **buffer_kwargs
            )


            
            path_getter = threading.Thread(target=get_paths, args=(path_queue,buffer))
            #batch_adder = threading.Thread(target=add_batch, args=(batch_queue,buffer))

            path_getter.start()
            #batch_adder.start()

            self.training_mode(True)
            while True:  
                    if buffer._size > 0:
                        start_train = time.time()
                        for _ in range(50):
                            start_sample = time.time()
                            train_data = buffer.random_batch(1000)
                            sam_time = time.time() - start_sample
                            print("Took to sample:", sam_time)
                            self.trainer.train(train_data)

                        train_time = time.time() - start_train
                        print("Took to train:", train_time, "\n")
                        tmp_policy = self.trainer._base_trainer.policy
                        tmp_policy.fuba += 1
                        collection_ns.policy = tmp_policy



