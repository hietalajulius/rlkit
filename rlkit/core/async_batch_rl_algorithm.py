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
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_rollouts_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            evaluation_data_collector,
            preset_evaluation_data_collector,
            num_collected_steps,
            buffer_memory_usage,
            collector_memory_usage,
            env_memory_usages,
            num_eval_param_buckets=1,
            save_policy_every_epoch=1,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            demo_paths=None,
            batch_queue=None,
            policy_weights_queue=None,
            new_policy_event=None,
            batch_processed_event=None,
            debug_same_batch=False
    ):
        super().__init__(
            trainer,
            evaluation_data_collector=evaluation_data_collector,
            preset_evaluation_data_collector=preset_evaluation_data_collector

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
        self.num_eval_rollouts_per_epoch = num_eval_rollouts_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.save_policy_every_epoch = save_policy_every_epoch
        self.num_eval_param_buckets = num_eval_param_buckets
        self.debug_same_batch = debug_same_batch

    def _train(self):
        start_time = time.time()
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        temp_policy_weights = copy.deepcopy(
            self.trainer._base_trainer.policy.state_dict())
        self.policy_weights_queue.put(temp_policy_weights)
        self.new_policy_event.set()

        print("Initialized policy")
        process = psutil.Process(os.getpid())

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if epoch % self.save_policy_every_epoch == 0:
                torch.save(
                    self.trainer._base_trainer.policy.state_dict(), 'async_policy/current_policy.mdl')
                torch.save(
                    self.trainer._base_trainer.alpha_optimizer.state_dict(), 'async_policy/current_alpha_optimizer.mdl')
                torch.save(
                    self.trainer._base_trainer.policy_optimizer.state_dict(), 'async_policy/current_policy_optimizer.mdl')
                torch.save(
                    self.trainer._base_trainer.qf1_optimizer.state_dict(), 'async_policy/current_qf1_optimizer.mdl')
                torch.save(
                    self.trainer._base_trainer.qf2_optimizer.state_dict(), 'async_policy/current_qf2_optimizer.mdl')

                print("Saved current policy")

            files = glob.glob('success_images/*')
            for f in files:
                os.remove(f)

            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_rollouts_per_epoch
            )

            self.preset_eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_param_buckets
            )
            gt.stamp('evaluation sampling')

            eval_paths = self.eval_data_collector.get_epoch_paths()
            print("EVAL SUCCESS RATRE", eval_util.get_generic_path_information(
                eval_paths)['env_infos/final/is_success Mean'])

            print("Epoch", epoch)
            for cycle in range(self.num_train_loops_per_epoch):

                # TODO: Use for memory debug
                #print("Memory usage in train",process.memory_info().rss/10E9, "GB")
                train_steps = epoch*self.num_train_loops_per_epoch * \
                    self.num_trains_per_train_loop + cycle*self.num_trains_per_train_loop

                while train_steps > self.train_collect_ratio * self.num_collected_steps.value:
                    print("Waiting collector to catch up...",
                          train_steps, self.num_collected_steps.value)
                    time.sleep(3)

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
                        #print("Updated policy")
                    if tren % 100 == 0:
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

                gt.stamp('training', unique=False)
                self.training_mode(False)
                self.trainer._base_trainer.policy.vers += 1

            self._end_epoch(epoch)
            print("Seconds since start", time.time() - start_time)
