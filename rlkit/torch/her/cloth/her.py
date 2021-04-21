import torch

from rlkit.torch.torch_rl_algorithm import TorchTrainer


class ClothDDPGHERTrainer(TorchTrainer):
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
            demo_data['observations'] = torch.cat(
                (demo_obs, demo_goals), dim=1)
            demo_data['next_observations'] = torch.cat(
                (demo_next_obs, demo_goals), dim=1)
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


class ClothSacHERTrainer(TorchTrainer):
    def __init__(self, base_trainer: TorchTrainer):
        super().__init__()
        self._base_trainer = base_trainer

    def train_from_torch(self, data):
        new_data = dict()
        new_data['rewards'] = data['rewards']
        new_data['terminals'] = data['terminals']
        new_data['actions'] = data['actions']
        

        resampled_goals = data['resampled_goals']

        value_obs = data['observations']
        value_next_obs = data['next_observations']

        if 'images' in data.keys():
            policy_obs = data['images']
            policy_next_obs = data['next_images']
            
        else:
            policy_obs = value_obs
            policy_next_obs = value_next_obs

        if 'robot_observations' in data.keys():
            policy_obs = torch.cat(
                (policy_obs, data['robot_observations']), axis=1)
            policy_next_obs = torch.cat(
                (policy_next_obs, data['next_robot_observations']), axis=1)
            value_obs = torch.cat(
                (value_obs, data['robot_observations']), axis=1)
            value_next_obs = torch.cat(
                (value_next_obs, data['next_robot_observations']), axis=1)
        if 'model_params' in data.keys():
            value_obs = torch.cat((value_obs, data['model_params']), axis=1)
            value_next_obs = torch.cat(
                (value_next_obs, data['model_params']), axis=1)

        policy_obs = torch.cat((policy_obs, resampled_goals), axis=1)
        policy_next_obs = torch.cat((policy_next_obs, resampled_goals), axis=1)
        value_obs = torch.cat((value_obs, resampled_goals), axis=1)
        value_next_obs = torch.cat((value_next_obs, resampled_goals), axis=1)


        new_data['policy_obs'] = policy_obs
        new_data['policy_next_obs'] = policy_next_obs
        new_data['value_obs'] = value_obs
        new_data['value_next_obs'] = value_next_obs

        new_data['corner_positions'] = data['corner_positions']

        self._base_trainer.train_from_torch(new_data)

    def get_diagnostics(self):
        return self._base_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self._base_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self._base_trainer.networks

    def get_snapshot(self):
        return self._base_trainer.get_snapshot()
