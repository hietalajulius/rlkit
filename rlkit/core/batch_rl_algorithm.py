import abc

import gtimer as gt
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
            num_eval_rollouts_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            preset_evaluation_data_collector: PathCollector = None,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            num_eval_param_buckets=1,
            save_policy_every_epoch=1,
            debug_same_batch=False
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            preset_evaluation_data_collector,
            replay_buffer,
        )
        # TODO: fix spaghetti
        self.reward_function = copy.deepcopy(replay_buffer.reward_function)

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

    def _train(self):
        load_existing = False  # TODO: parametrize
        if load_existing:
            self.trainer._base_trainer.policy.load_state_dict(torch.load(
                'policy/current_policy.mdl'))
            self.trainer._base_trainer.alpha_optimizer.load_state_dict(
                torch.load('policy/current_alpha_optimizer.mdl'))
            self.trainer._base_trainer.policy_optimizer.load_state_dict(
                torch.load('policy/current_policy_optimizer.mdl'))
            self.trainer._base_trainer.qf1_optimizer.load_state_dict(
                torch.load('policy/current_qf1_optimizer.mdl'))
            self.trainer._base_trainer.qf2_optimizer.load_state_dict(
                torch.load('policy/current_qf2_optimizer.mdl'))
            with open('policy/buffer_data.pkl', 'rb') as inp:
                self.replay_buffer = pickle.load(inp)
                self.replay_buffer.set_reward_function(self.reward_function)
                print("LOADED BUFFER", self.replay_buffer._size)
            print("LOADED POLICY")

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

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if epoch % self.save_policy_every_epoch == 0:
                torch.save(
                    self.trainer._base_trainer.policy.state_dict(), 'policy/current_policy.mdl')
                torch.save(
                    self.trainer._base_trainer.alpha_optimizer.state_dict(), 'policy/current_alpha_optimizer.mdl')
                torch.save(
                    self.trainer._base_trainer.policy_optimizer.state_dict(), 'policy/current_policy_optimizer.mdl')
                torch.save(
                    self.trainer._base_trainer.qf1_optimizer.state_dict(), 'policy/current_qf1_optimizer.mdl')
                torch.save(
                    self.trainer._base_trainer.qf2_optimizer.state_dict(), 'policy/current_qf2_optimizer.mdl')

                self.replay_buffer.set_reward_function(None)
                with open('policy/buffer_data.pkl', 'wb') as outp:
                    pickle.dump(self.replay_buffer, outp,
                                pickle.HIGHEST_PROTOCOL)
                    print("DUMPED buffer")
                self.replay_buffer.set_reward_function(self.reward_function)
                print("SAVED CURRENT POLICY")

            files = glob.glob('success_images/*')
            for f in files:
                os.remove(f)
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_rollouts_per_epoch
            )
            gt.stamp('evaluation sampling')

            eval_paths = self.eval_data_collector.get_epoch_paths()
            print("EVAL SUCCESS RATRE", eval_util.get_generic_path_information(
                eval_paths)['env_infos/final/is_success Mean'])

            if not self.preset_eval_data_collector is None:
                self.preset_eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_param_buckets
                )
                preset_eval_paths = self.preset_eval_data_collector.get_epoch_paths()
                print("PRESET_EVAL SUCCESS RATRE", eval_util.get_generic_path_information(
                    preset_eval_paths)['env_infos/final/is_success Mean'])

            print("Epoch", epoch)

            for cycle in range(self.num_train_loops_per_epoch):
                print("Cycle", cycle)
                start_cycle = time.time()

                if not self.debug_same_batch:
                    new_expl_paths = self.expl_data_collector.collect_new_paths(
                        self.max_path_length,
                        self.num_expl_steps_per_train_loop,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)
                    collection_done = time.time()
                    collection_time = collection_done - start_cycle
                    print("Took to collect", collection_time)

                    self.replay_buffer.add_paths(new_expl_paths)
                    gt.stamp('data storing', unique=False)

                self.training_mode(True)
                train_start = time.time()
                for _ in range(self.num_trains_per_train_loop):
                    if not self.debug_same_batch:
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)
                train_time = time.time() - train_start
                print("Took to train", train_time)
            self._end_epoch(epoch)
            print("Seconds since start", time.time() - start_time)
