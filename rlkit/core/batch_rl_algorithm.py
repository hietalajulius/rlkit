import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
import time
import torch


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
                print("Took to train:", col_time)
                train_times += tra_time
                gt.stamp('training', unique=False)
                self.training_mode(False)
                total_times += time.time() - start_collect
            print("Time collect avg cycle:", collect_times/self.num_train_loops_per_epoch)
            print("Time train avg cycle:", train_times/self.num_train_loops_per_epoch)
            print("Total avg cycle:", total_times/self.num_train_loops_per_epoch)
            print("Ending epoch")
            self._end_epoch(epoch)
