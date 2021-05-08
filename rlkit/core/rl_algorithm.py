import abc
from collections import OrderedDict

import gtimer as gt

from rlkit.core import logger, eval_util
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import DataCollector

from torch.utils.tensorboard import SummaryWriter
import numpy as np


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env=None,
            evaluation_env=None,
            exploration_data_collector: DataCollector = None,
            evaluation_data_collector: DataCollector = None,
            preset_evaluation_data_collector: DataCollector = None,
            demo_data_collector: DataCollector = None,
            replay_buffer: ReplayBuffer = None,
            demo_buffer: ReplayBuffer = None,
            demo_paths=None
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.preset_eval_data_collector = preset_evaluation_data_collector
        self.demo_data_collector = demo_data_collector
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self._start_epoch = 0
        self.demo_paths = demo_paths

        self.post_epoch_funcs = []

        self.writer = SummaryWriter(log_dir='tblogs/'+logger._prefixes[0])

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self, epoch, save_params=False):
        snapshot = self._get_snapshot()
        # TODO: figure out below, only log tb for now
        # if save_params:
        #logger.save_itr_params(epoch, snapshot)
        # gt.stamp('saving')
        # self._log_stats(epoch)

        self._log_tb_stats(epoch)

        if not self.expl_data_collector is None:
            self.expl_data_collector.end_epoch(epoch)
        if not self.eval_data_collector is None:
            self.eval_data_collector.end_epoch(epoch)
        if not self.preset_eval_data_collector is None:
            self.preset_eval_data_collector.end_epoch(epoch)
        if not self.replay_buffer is None:
            self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v

        if not self.expl_data_collector is None:
            for k, v in self.expl_data_collector.get_snapshot().items():
                snapshot['exploration/' + k] = v

        if not self.eval_data_collector is None:
            for k, v in self.eval_data_collector.get_snapshot().items():
                snapshot['evaluation/' + k] = v

        if not self.preset_eval_data_collector is None:
            for k, v in self.preset_eval_data_collector.get_snapshot().items():
                snapshot['preset_evaluation/' + k] = v
        if not self.replay_buffer is None:
            for k, v in self.replay_buffer.get_snapshot().items():
                snapshot['replay_buffer/' + k] = v
        return snapshot

    def _log_tb_stats(self, epoch):

        expl_paths = self.expl_data_collector.get_epoch_paths()
        eval_paths = self.eval_data_collector.get_epoch_paths()

        self.writer.add_scalar('test_regular/eval', eval_util.get_generic_path_information(
            eval_paths)['env_infos/final/is_success Mean'], epoch)

        '''
        ates = []
        for path in expl_paths:
            ate = np.sqrt(eval_util.get_generic_path_information([path])[
                          'env_infos/squared_error_norm Mean'])
            ates.append(ate)
        ates = np.array(ates)
        '''

        self.writer.add_scalar('expl/delta_size_penalty', eval_util.get_generic_path_information(expl_paths)[
                          'env_infos/delta_size_penalty Mean'], epoch)

        self.writer.add_scalar('expl/ate_penalty', eval_util.get_generic_path_information(expl_paths)[
                          'env_infos/ate_penalty Mean'], epoch)

        self.writer.add_scalar('expl/task_reward', eval_util.get_generic_path_information(expl_paths)[
                          'env_infos/task_reward Mean'], epoch)

        self.writer.add_scalar('expl/control_penalty', eval_util.get_generic_path_information(expl_paths)[
                          'env_infos/control_penalty Mean'], epoch)

        self.writer.add_scalar('expl/cosine_similarity', eval_util.get_generic_path_information(expl_paths)[
                          'env_infos/cosine_similarity Mean'], epoch)

        self.writer.add_scalar('expl/reward', eval_util.get_generic_path_information(expl_paths)[
                          'Rewards Mean'], epoch)

        self.writer.add_scalar('expl/returns', eval_util.get_generic_path_information(expl_paths)[
                          'Returns Mean'], epoch)
        

        if not self.preset_eval_data_collector is None:
            preset_eval_paths = self.preset_eval_data_collector.get_epoch_paths()
            self.writer.add_scalar('test_preset/eval', eval_util.get_generic_path_information(
                preset_eval_paths)['env_infos/final/is_success Mean'], epoch)

        if 'State estimation loss' in self.trainer.get_diagnostics().keys():
            self.writer.add_scalar(
                'losses/eval/state', self.trainer.get_diagnostics()['State estimation loss'], epoch)

        self.writer.add_scalar(
                'losses/eval/q1', self.trainer.get_diagnostics()['QF1 Loss'], epoch)
        self.writer.add_scalar(
                'losses/eval/q2', self.trainer.get_diagnostics()['QF2 Loss'], epoch)
        self.writer.add_scalar(
                'losses/eval/policy', self.trainer.get_diagnostics()['Raw Policy Loss'], epoch)
        self.writer.add_scalar(
                'losses/eval/logpi', self.trainer.get_diagnostics()['Log Pi'], epoch)
        self.writer.add_scalar(
                'losses/eval/alpha', self.trainer.get_diagnostics()['Alpha'], epoch)
        self.writer.add_scalar(
                'losses/eval/alphaloss', self.trainer.get_diagnostics()['Alpha Loss'], epoch)

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """

        if not self.replay_buffer is None:
            logger.record_dict(
                self.replay_buffer.get_diagnostics(),
                prefix='replay_buffer/'
            )

        """
        Trainer
        """
        logger.record_dict(self.trainer.get_diagnostics(), prefix='trainer/')

        """
        Exploration
        """
        if not self.expl_data_collector is None:
            logger.record_dict(
                self.expl_data_collector.get_diagnostics(),
                prefix='exploration/'
            )
            expl_paths = self.expl_data_collector.get_epoch_paths()
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        if not self.expl_env is None:
            if hasattr(self.expl_env, 'get_diagnostics'):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix='exploration/',
                )

        """
        Evaluation
        """
        if not self.eval_data_collector is None:
            logger.record_dict(
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = self.eval_data_collector.get_epoch_paths()
            logger.record_dict(
                eval_util.get_generic_path_information(eval_paths),
                prefix="evaluation/",
            )

        if not self.preset_eval_data_collector is None:
            logger.record_dict(
                self.preset_eval_data_collector.get_diagnostics(),
                prefix='preset_evaluation/',
            )
            preset_eval_paths = self.preset_eval_data_collector.get_epoch_paths()
            logger.record_dict(
                eval_util.get_generic_path_information(preset_eval_paths),
                prefix="preset_evaluation/",
            )

        if not self.eval_env is None:
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/',
                )

        if not self.demo_paths is None:
            print("Logging demo path stats")
            logger.record_dict(
                eval_util.get_generic_path_information(self.demo_paths),
                prefix="evaluation/demonstrations/",
            )

        """
        Misc
        """
        gt.stamp('logging')

        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
