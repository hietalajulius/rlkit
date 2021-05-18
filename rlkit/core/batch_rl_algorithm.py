import abc

from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.core import eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import time
import glob
import os
import torch
import pickle
import copy
import psutil

class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            save_folder,
            title,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_rollouts_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            preset_evaluation_data_collector: PathCollector = None,
            demo_data_collector: PathCollector = None,
            num_demos= 0,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            num_eval_param_buckets=1,
            save_policy_every_epoch=1,
            debug_same_batch=False,
            script_policy=None
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            preset_evaluation_data_collector,
            demo_data_collector,
            replay_buffer,
        )
        # TODO: fix spaghetti
        self.task_reward_function = copy.deepcopy(
            replay_buffer.task_reward_function)

        self.num_demos = num_demos

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_rollouts_per_epoch = num_eval_rollouts_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.num_eval_param_buckets = num_eval_param_buckets
        self.save_policy_every_epoch = save_policy_every_epoch
        self.debug_same_batch = debug_same_batch
        self.title = title
        self.save_folder = save_folder
        self.script_policy = script_policy

    def _train(self):
        self.training_mode(True)
        if not self.demo_data_collector is None:
            print("Collecting demos:", self.num_demos*self.max_path_length)
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

            if self.debug_same_batch:
                train_data = self.replay_buffer.random_batch(
                    self.batch_size)

        for epoch in range(self._start_epoch, self.num_epochs):
            if epoch % self.save_policy_every_epoch == 0:
                items_in_policy_dir = len(os.listdir(f'{self.save_folder}/policies'))/2
                policy_path = f'{self.save_folder}/policies/policy_{items_in_policy_dir}'
                torch.save(self.trainer._base_trainer.policy.state_dict(), f'{policy_path}.mdl')
                print(f"Saved {policy_path}.mdl")

                if not self.script_policy is None:
                    self.script_policy.load_state_dict(torch.load(f'{policy_path}.mdl', map_location='cpu'))
                    self.script_policy.eval()
                    sm = torch.jit.script(self.script_policy).cpu()
                    torch.jit.save(sm, f'{policy_path}.pt')
                    print(f"Saved {policy_path}.pt")
                

            files = glob.glob('success_images/*')
            for f in files:
                os.remove(f)
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_rollouts_per_epoch
            )

            if not self.preset_eval_data_collector is None:
                self.preset_eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_param_buckets
                )

            print("Epoch", epoch)

            for cycle in range(self.num_train_loops_per_epoch):
                print("Cycle", cycle, epoch)
                #print("Memory usage in main process", process.memory_info().rss/1E9)
                start_cycle = time.time()
                self.training_mode(True)
                if not self.debug_same_batch:
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
                    if not self.debug_same_batch:
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                    self.trainer.train(train_data)
                self.training_mode(False)
                train_time = time.time() - train_start
                print("Took to train", train_time, "Time:", time.asctime())
            self._end_epoch(epoch)
            print("Seconds since start", time.time() - start_time)
