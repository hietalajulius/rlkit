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
from queue import LifoQueue
import threading
from rlkit.torch.sac.policies import TanhGaussianPolicy
import copy
import os
import multiprocessing
import copy


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
                collect_times += col_time
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                print("Took to collect:", col_time, self.replay_buffer._size)

                gt.stamp('data storing', unique=False)

                start_train = time.time()
                self.training_mode(True)
                sam_times_cycle = 0
                train_train_times_cycle = 0
                
                for tren in range(self.num_trains_per_train_loop):
                    start_sam = time.time()
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    sam_time = time.time() - start_sam
                    sam_times_cycle += sam_time
                    #print("Took to sample:", sam_time)
                    if not self.demo_buffer == None:
                        demo_data = self.demo_buffer.random_batch(int(self.batch_size*(1/8)))
                        self.trainer.train(train_data, demo_data)
                    else:
                        start_train_train = time.time()
                        self.trainer.train(train_data)
                        train_train_time = time.time() - start_train_train
                        train_train_times_cycle += train_train_time
                tra_time = time.time() - start_train
                print("Took to train: \n",
                        tra_time,
                        "\nAverage pure train: \n",
                        train_train_times_cycle/self.num_trains_per_train_loop,
                        "\nAverage sample time: \n",
                        sam_times_cycle/self.num_trains_per_train_loop,
                        "\nAverage full sample and train: \n",
                        tra_time/self.num_trains_per_train_loop
                        )
                train_times += tra_time
                gt.stamp('training', unique=False)
                self.training_mode(False)
                total_times += time.time() - start_collect
            print("Time collect avg cycle:", collect_times/self.num_train_loops_per_epoch)
            print("Time train avg cycle:", train_times/self.num_train_loops_per_epoch)
            print("Total avg cycle:", total_times/self.num_train_loops_per_epoch)
            print("Ending epoch")
            self._end_epoch(epoch)



def get_paths(new_path_queue, batch_queue, env_kwargs, buffer_kwargs):
    print("Get paths started")
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        del os.environ['CUDA_VISIBLE_DEVICES']
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
    buffer_env = NormalizedBoxEnv(gym.make('Cloth-v1', **env_kwargs))
    buffer = ObsDictRelabelingBuffer(
            env=buffer_env,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            **buffer_kwargs
    )
    

    while True:
        for _ in range(2):
            new_path = new_path_queue.get()
            buffer.add_path(new_path)
        for _ in range(100):
            batch = buffer.random_batch(256)
            batch_queue.put(batch)
            


def add_batch(batch_queue, buffer):
    while True:
        if buffer._size > 0:
            batch = buffer.random_batch(1000)
            #print("Put batch to queue")
            batch_queue.put(batch)


def path_collector_process(path_queue, namespace, index, env_kwargs, desired_goal_key, observation_key, additional_keys):
    if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
        del os.environ['CUDA_VISIBLE_DEVICES']
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps"
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log"
    env = NormalizedBoxEnv(gym.make('Cloth-v1', **env_kwargs))
    print("Env was created")

    def obs_processor(o):
        obs = o[observation_key]
        for key in additional_keys:
            obs = np.hstack((obs, o[key]))
        obs = np.hstack((obs, o[desired_goal_key]))
        return obs

    print("start path collection in idx", index)
    while True:
        current_policy = namespace.policy
        #print("Policy version", current_policy.vers)
        path = rollout(env, current_policy, max_path_length=50, preprocess_obs_for_policy_fn=obs_processor)
        print("Collected a path in idx", index)
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
            namespace=None,
            batch_queue=None
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
        self.namespace = namespace
        self.batch_queue = batch_queue
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

            print("Epoch", epoch)
            collect_times = 0
            train_times = 0
            total_times = 0
            for cycle in range(self.num_train_loops_per_epoch):
                #print("Namespa", self.namespace)
                #print("\n Cycle", cycle, epoch)
                start_collect = time.time()
                '''
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                col_time = time.time() - start_collect
                collect_times += col_time
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                print("Took to collect:", col_time, self.replay_buffer._size)

                gt.stamp('data storing', unique=False)
                '''

                start_train = time.time()
                self.training_mode(True)
                sam_times_cycle = 0
                train_train_times_cycle = 0
                '''
                train_data = self.replay_buffer.random_batch( #Batches need to be gotten faster
                        self.batch_size)
                '''
                temp_policy = copy.deepcopy(self.trainer._base_trainer.policy)
                temp_policy.to('cpu')
                self.namespace.policy = temp_policy
                print("Updated policy")
                for tren in range(self.num_trains_per_train_loop):
                    start_sam = time.time()
                    print("Getting from batch q", self.batch_queue.qsize())
                    train_data = self.batch_queue.get()
                    print("Got from batch q", self.batch_queue.qsize())
                    sam_time = time.time() - start_sam
                    sam_times_cycle += sam_time
                    #print("Took to sample:", sam_time)
                    start_train_train = time.time()
                    self.trainer.train(train_data)
                    train_train_time = time.time() - start_train_train
                    train_train_times_cycle += train_train_time
                tra_time = time.time() - start_train
                
                print("Took to train: \n",
                        tra_time,
                        "\nAverage pure train: \n",
                        train_train_times_cycle/self.num_trains_per_train_loop,
                        "\nAverage sample time: \n",
                        sam_times_cycle/self.num_trains_per_train_loop,
                        "\nAverage full sample and train: \n",
                        tra_time/self.num_trains_per_train_loop
                        )
                
                train_times += tra_time
                gt.stamp('training', unique=False)
                self.training_mode(False)
                total_times += time.time() - start_collect
                self.trainer._base_trainer.policy.vers += 1
                
            print("Time collect avg cycle:", collect_times/self.num_train_loops_per_epoch)
            print("Time train avg cycle:", train_times/self.num_train_loops_per_epoch)
            print("Total avg cycle:", total_times/self.num_train_loops_per_epoch)
            print("Ending epoch")