import abc

from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.core import eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.samplers.eval_suite.utils import create_base_epoch_directory
import time
import glob
import os
import torch
import pickle
import copy
import psutil
import tracemalloc

def dump_models(save_folder, epoch, trainer, script_policy=None):
    policy_state_dict = trainer.policy.state_dict()
    torch.save(policy_state_dict, f'{save_folder}/current_policy.mdl')
    torch.save(trainer.alpha_optimizer.state_dict(), f'{save_folder}/current_alpha_optimizer.mdl')
    torch.save(trainer.policy_optimizer.state_dict(), f'{save_folder}/current_policy_optimizer.mdl')
    torch.save(trainer.qf1_optimizer.state_dict(), f'{save_folder}/current_qf1_optimizer.mdl')
    torch.save(trainer.qf2_optimizer.state_dict(), f'{save_folder}/current_qf2_optimizer.mdl')

    if not script_policy is None:
        script_policy.load_state_dict(policy_state_dict)
        script_policy.eval()
        sm = torch.jit.script(script_policy).cpu()
        torch.jit.save(sm, f'{save_folder}/epochs/{epoch}/policy.pt')

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            save_folder,
            batch_size,
            max_path_length,
            num_epochs,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            demo_data_collector: PathCollector = None,
            num_demos= 0,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_policy_every_epoch=1,
            script_policy=None,
            eval_suite=None
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            None,
            replay_buffer,
        )
        # TODO: fix spaghetti
        self.task_reward_function = copy.deepcopy(
            replay_buffer.task_reward_function)

        self.num_demos = num_demos

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.save_policy_every_epoch = save_policy_every_epoch
        self.save_folder = save_folder
        self.script_policy = script_policy
        self.eval_suite = eval_suite
        self.demo_data_collector = demo_data_collector

    def _train(self):
        current_memory_snapshot = tracemalloc.take_snapshot()
        first_memory_snapshot = copy.deepcopy(current_memory_snapshot)
        self.training_mode(True)
        if not self.demo_data_collector is None:
            print("Collecting demos:", self.num_demos)
            demo_paths = self.demo_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_demos*self.max_path_length,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(demo_paths)

        start_time = time.time()

        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        for epoch in range(self._start_epoch, self.num_epochs):
            create_base_epoch_directory(self.save_folder, epoch)
            self.training_mode(True)    
            print("Epoch", epoch)

            for cycle in range(self.num_train_loops_per_epoch):
                print("Cycle", cycle, epoch)
                
                new_memory_snapshot = tracemalloc.take_snapshot()
                top_stats_current = new_memory_snapshot.compare_to(current_memory_snapshot, 'lineno')
                top_stats_first = new_memory_snapshot.compare_to(first_memory_snapshot, 'lineno')
                print("[ Top 10 differences to current ]")
                for stat in top_stats_current[:10]:
                    print(stat)
                print("\n")
                print("[ Top 10 differences to fisrt ]")
                for stat in top_stats_first[:10]:
                    print(stat)
                print("\n")

                current_memory_snapshot = new_memory_snapshot
                start_cycle = time.time()
                
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                collection_done = time.time()
                collection_time = collection_done - start_cycle

                self.replay_buffer.add_paths(new_expl_paths)
                print("Took to collect:", collection_time, "buffer size:", self.replay_buffer._size)
                train_start = time.time()
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                
                train_time = time.time() - train_start
                print("Took to train", train_time, "Time:", time.asctime())

            self.training_mode(False)

            if epoch % self.save_policy_every_epoch == 0:
                dump_models(self.save_folder, epoch, self.trainer._base_trainer, self.script_policy)
                print("Dumped models")

            eval_stats = self.eval_suite.run_evaluations(epoch)
            self.log_test_suite_tb_stats(epoch, eval_stats)
            self._end_epoch(epoch)
            print("Seconds since start", time.time() - start_time)
