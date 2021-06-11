import typing
from typing import List
from rlkit.samplers.eval_suite.utils import create_base_epoch_directory
from torch.serialization import save


class EvalTest(object):
    def __init__(self, env, policy, name, metric_keys, num_runs, variant):
        self.env = env
        self.policy = policy
        self.num_runs = num_runs
        self.name = name
        self.base_save_folder = variant['save_folder']
        self.epoch = 0
        self.metric_keys = metric_keys

    def run_evaluations(self) -> float:
        results = dict()
        for metric_key in self.metric_keys:
            results[metric_key] = 0

        for i in range(self.num_runs):
            result = self.single_evaluation(i)
            for result_key in result.keys():
                results[result_key] += result[result_key]/self.num_runs

        return results

    def single_evaluation(self) -> float:
        raise NotImplementedError("You need to implement single_evaluation")

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class EvalTestSuite(object):
    def __init__(self, tests: List[EvalTest], save_folder):
        self.tests = tests
        self.base_save_folder = save_folder

    def add_evaluation_test(self, test) -> None:
        self.tests.append(test)

    def run_evaluations(self, epoch: int) -> dict:
        results = dict()
        for test in self.tests:
            test.set_epoch(epoch)
            metrics = test.run_evaluations()
            for metric_key in metrics.keys():
                results[f"{test.name}_{metric_key}"] = metrics[metric_key]
        return results
    