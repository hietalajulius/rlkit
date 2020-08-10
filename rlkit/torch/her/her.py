import torch

from rlkit.torch.torch_rl_algorithm import TorchTrainer


class HERTrainer(TorchTrainer):
    def __init__(self, base_trainer: TorchTrainer):
        super().__init__()
        self._base_trainer = base_trainer

    def train_from_torch(self, data, demo_data=None):
        obs = data['observations']
        next_obs = data['next_observations']
        goals = data['resampled_goals']
        data['observations'] = torch.cat((obs, goals), dim=1)
        data['next_observations'] = torch.cat((next_obs, goals), dim=1)
        if not demo_data == None:
            demo_obs = demo_data['observations']
            demo_next_obs = demo_data['next_observations']
            demo_goals = demo_data['resampled_goals']
            demo_data['observations'] = torch.cat((demo_obs, demo_goals), dim=1)
            demo_data['next_observations'] = torch.cat((demo_next_obs, demo_goals), dim=1)
            self._base_trainer.train_from_torch(data, demo_data)
        else:
            self._base_trainer.train_from_torch(data)

    def get_diagnostics(self):
        return self._base_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self._base_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self._base_trainer.networks

    def get_snapshot(self):
        return self._base_trainer.get_snapshot()
