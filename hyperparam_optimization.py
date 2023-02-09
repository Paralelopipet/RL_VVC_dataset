from collections import defaultdict
import pickle
from sklearn import metrics
from vvc import offline_vvc, online_vvc, test_vvc_verbose, deleteAllTensorboardFiles
from plot import plot_res1, plot_res2
from datetime import datetime
from enum import Enum


from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import os


def train_vvc(config, checkpoint_dir=None, data_dir=None):
    os.chdir(data_dir)
    config['seed'] = seed
    offline_res = offline_vvc(config, data_dir)
    online_res, test_res = online_vvc(config, offline_res)

    training_set_reward_diff = sum(online_res['average_reward_diff (r - rbaseline)'])
    test_set_reward_diff = sum(test_res['average_reward_diff (r - rbaseline)'])
    print(training_set_reward_diff)
    print(test_set_reward_diff)

    tune.report(training_set_reward_diff=training_set_reward_diff, test_set_reward_diff=test_set_reward_diff)


envs = ['123']
#envs = ['13']
algos = ['wcsac']
# = ['sac']
# seeds = [0, 1, 2]
seed = 0

# save timestamp
timestamp = True
# time stamp
now = ''
if timestamp:
    now = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
else:
    now = ""


class RewardOption(Enum):
    SWITCHINGLINEARVOLTCIRCUIT = 1
    SWITCHINGLINEARVOLT = 2
    SWITCHINGDISCRETEVOLTCIRCUIT = 3
    SWITCHINGDISCRETEVOLT = 4
    CONSTRAINTNOSWITCHING = 5
    DISCRETEVOLTCIRCUIT = 6


deleteAllTensorboardFiles()
for env in envs:
    for algo in algos:
        config = {
            "env": env,
            "state_option": 2,
            "reward_option": RewardOption.DISCRETEVOLTCIRCUIT.value,
            "offline_split": 0.1,  # initial values
            "online_split": 0.8,
            "test_split": 0.1,
            "replay_size": 3000,
            "test_result": 10,
            "seed": 0,
        }
        if algo == 'wcsac':
            config['algo'] = {
                "algo": "wcsac",
                "dims_hidden_neurons": tune.choice([ (32, 32), (64, 64), (128, 128), (256, 256)]),
                "scale_reward": 5.0,
                "discount": 0.95,
                "alpha": 0.1,
                "batch_size": tune.choice([32, 64, 128, 256]),
                "lr": tune.loguniform(1e-5, 1e-2),
                "smooth": 0.99,
                "offline_training_steps": 100,
                "online_training_steps": 10,
                "max_episode_len": 1000,
                "damp_scale": 0.1,  # 0 for not in use, 10 in original algorithm
                "cost_limit": 15,  # 15 in original algo, eq 10, parameter d
                "init_temperature": 0.399,
                "betas": [0.9, 0.999],
                "lr_scale": 1
            }
            config['reward_option'] = RewardOption.CONSTRAINTNOSWITCHING.value
        else:
            break

res = defaultdict(list)
# The functionality of both dictionaries and defaultdict are almost same except for the fact that defaultdict
# never raises a KeyError. It provides a default value for the key that does not exists like []

tune.TuneConfig(max_concurrent_trials=1)

scheduler = ASHAScheduler(
    metric="test_set_reward_diff",
    mode="max",
    max_t=2,
    grace_period=1,
    reduction_factor=2)
reporter = CLIReporter(
    # parameter_columns=["l1", "l2", "lr", "batch_size"],
    metric_columns=["training_set_reward_diff", "test_set_reward_diff"])
result = tune.run(
    partial(train_vvc, data_dir=os.path.dirname(__file__)),
    resources_per_trial={"cpu": 1, "gpu": 0},
    config=config,
    max_concurrent_trials=1,
    num_samples=20,
    scheduler=scheduler,
    progress_reporter=reporter)

best_trial = result.get_best_trial("test_set_reward_diff", "max", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final training_set_reward_diff: {}".format(
    best_trial.last_result["training_set_reward_diff"]))
print("Best trial final test_set_reward_diff: {}".format(
    best_trial.last_result["test_set_reward_diff"]))







