import abc
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.core import eval_util
import time
import copy
import os
import psutil
import glob
import torch


class AsyncBatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            save_folder,
            batch_size,
            max_path_length,
            num_epochs,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_collected_steps,
            buffer_memory_usage,
            collector_memory_usage,
            env_memory_usages,
            demo_data_collector = None,
            num_demos= 0,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_policy_every_epoch=1,
            script_policy=None,
            eval_suite=None,
            batch_queue=None,
            policy_weights_queue=None,
            new_policy_event=None,
            batch_processed_event=None,
    ):
        super().__init__(
            trainer,
            None,
            None,
            None,
            None,
            None,
        )
        self.train_collect_ratio = 4
        self.num_collected_steps = num_collected_steps
        self.buffer_memory_usage = buffer_memory_usage
        self.collector_memory_usage = collector_memory_usage
        self.env_memory_usages = env_memory_usages
        self.batch_queue = batch_queue
        self.policy_weights_queue = policy_weights_queue
        self.new_policy_event = new_policy_event
        self.batch_processed_event = batch_processed_event
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.save_policy_every_epoch = save_policy_every_epoch

        self.num_demos = num_demos

        self.save_folder = save_folder
        self.script_policy = script_policy
        self.eval_suite = eval_suite
        self.demo_data_collector = demo_data_collector


    def _train(self):
        start_time = time.time()
        temp_policy_weights = copy.deepcopy(
            self.trainer._base_trainer.policy.state_dict())
        self.policy_weights_queue.put(temp_policy_weights)
        self.new_policy_event.set()

        print("Initialized policy")
        process = psutil.Process(os.getpid())

        for epoch in range(self._start_epoch, self.num_epochs):
            print("Epoch", epoch)
            for cycle in range(self.num_train_loops_per_epoch):
                train_steps = epoch*self.num_train_loops_per_epoch * \
                    self.num_trains_per_train_loop + cycle*self.num_trains_per_train_loop

                while train_steps > self.train_collect_ratio * self.num_collected_steps.value:
                    print("Waiting collectors to catch up...",
                          train_steps, self.num_collected_steps.value)
                    time.sleep(10)

                start_cycle = time.time()
                self.training_mode(True)
                sam_times_cycle = 0
                train_train_times_cycle = 0

                for tren in range(self.num_trains_per_train_loop):
                    start_sam = time.time()
                    train_data = self.batch_queue.get()
                    self.batch_processed_event.set()

                    sam_time = time.time() - start_sam
                    sam_times_cycle += sam_time

                    start_train_train = time.time()
                    self.trainer.train_from_torch(train_data)
                    del train_data

                    train_train_time = time.time() - start_train_train

                    if not self.new_policy_event.is_set():
                        temp_policy_weights = copy.deepcopy(
                            self.trainer._base_trainer.policy.state_dict())
                        self.policy_weights_queue.put(temp_policy_weights)
                        self.new_policy_event.set()
                        print("Algo: updated policy", tren)
                    if tren % 200 == 0:
                        print("--STATUS--")
                        print(tren, "/", self.num_trains_per_train_loop,
                              "Took to sample:", sam_time)
                        print(tren, "/", self.num_trains_per_train_loop,
                              "Took to train:", train_train_time)
                        print("Total train steps so far:", epoch*self.num_train_loops_per_epoch *
                              self.num_trains_per_train_loop + cycle*self.num_trains_per_train_loop + tren)
                        print("Total collected steps in train",
                              self.num_collected_steps.value)
                        print("Memory usages, train:",
                              process.memory_info().rss/10E9, "buffer:", self.buffer_memory_usage.value, "collector:", self.collector_memory_usage.value, "envs:", [emu.value for emu in self.env_memory_usages], "\n")

                    train_train_times_cycle += train_train_time

                cycle_time = time.time() - start_cycle

                print("Cycle", cycle, "took: \n",
                      cycle_time,
                      "\nAverage pure train: \n",
                      train_train_times_cycle/self.num_trains_per_train_loop,
                      "\nAverage sample time: \n",
                      sam_times_cycle/self.num_trains_per_train_loop,
                      "\nAverage full sample and train: \n",
                      cycle_time/self.num_trains_per_train_loop
                      )

                self.training_mode(False)

            self._end_epoch(epoch)
            print("Seconds since start", time.time() - start_time)
