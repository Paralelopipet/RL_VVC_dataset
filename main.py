from collections import defaultdict
import pickle

from sklearn import metrics
from vvc import offline_vvc, online_vvc
from plot import plot_res
from datetime import datetime

# envs = ['13', '123', '8500']
envs = ['13']
# algos = ['dqn', 'sac']
algos = ['sac']
#seeds = [0, 1, 2]
seeds = [0]

# save timestamp
timestamp = True
# time stamp
now = [];
for env in envs:
    for algo in algos:
        config = {
                "env": env,
                "state_option": 2,
                "reward_option": 1,
                "offline_split": 0.1,
                "online_split": 0.1,
                "test_split": 0.1,
                "replay_size": 3000,
                "seed": 0,
        }

        if algo == 'sac':
            config['algo'] = {
                "algo": "sac",
                "dims_hidden_neurons": (120, 120),
                "scale_reward": 5.0,
                "discount": 0.95,
                "alpha": .2,
                "batch_size": 64,
                "lr": 0.0005,
                "smooth": 0.99,
                "offline_training_steps": 100,
                "online_training_steps": 2,
            }
        elif algo == 'dqn':
            config['algo'] = {
                "algo": "dqn",
                "dims_hidden_neurons": (120, 120),
                "scale_reward": 5.0,
                "discount": 0.95,
                "batch_size": 64,
                "lr": 0.0005,
                "copy_steps": 30,
                "eps_len": 500,
                "eps_max": 1.0,
                "eps_min": 0.02,
                "offline_training_steps": 100,
                "online_training_steps": 2,
            }
        else:
            break

        res = defaultdict(list)
        for seed in seeds:
            config['seed'] = seed
            offline_res = offline_vvc(config)
            online_res, test_res = online_vvc(config, offline_res)

            # online results
            for k, v in online_res.items():
                res['online_' + k].append(v)
            # test results
            for k, v in test_res.items():
                res['test_' + k].append(v)
        
        if timestamp:
            dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        else:
            dt_string = ""
        now.append(dt_string)
        with open('./res/data/{}_{}{}.pkl'.format(config['env'],
                                                config['algo']['algo'],
                                                dt_string), 'wb') as f:
            pickle.dump(res, f)

smoothing = 100  # smooth the curves by moving average # default_smooth = 20
# metric = 'online_reward_diff (r - rbaseline)'
metrics = ['max voltage violation', 'average max voltage violation', 'reward_diff (r - rbaseline)', 'average_reward_diff (r - rbaseline)']

ylabel={'reward_diff (r - rbaseline)': 'Reward(RL) - Reward(baseline)',
        'average_reward_diff (r - rbaseline)': 'Average Reward(RL) - Reward(baseline)',
        'max voltage violation': 'Maximum voltage violation (volt)',
        'average max voltage violation': 'Average Maximum voltage violation (volt)'}

for metric in metrics:
    
    if 'average' in metric:
        smooth_param = 1
    else:
        smooth_param = smoothing
    plot_res(envs=['13', '123', '8500'],
            algos=algos,
            metric=metric,
            smoothing=smooth_param,
            ylabel=ylabel[metric],
            xlabel='Timestamp (half-hour)',
            time_stamps = now)