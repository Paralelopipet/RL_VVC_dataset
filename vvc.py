import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs.env import VVCEnv
import copy
from enum import Enum
from algos.replay import ReplayBuffer


class Mode(Enum):
    OFFLINE = 1
    ONLINE = 2
    TEST = 3


def _env_setup(config):
    return VVCEnv(config['env'],
                  config['state_option'],
                  config['reward_option'],
                  config['offline_split'],
                  config['online_split'],
                  config['test_split'])


def _agent_setup(config, env):
    try:
        module = importlib.import_module('algos.{}'.format(config['algo']['algo']))
        Agent = getattr(module, 'Agent')
    except ImportError:
        raise ImportError('Algorithm {} not found'.format(config['algo']['algo']))
    return Agent(config, env)


def _data2replay(env, replay, scale_reward):
    for iter in tqdm(range(env.len_offline), desc="Converting data to transition tuples"):
        s = env.state
        s_next, reward, done, info = env.step()
        baseline_action = info['action']

        replay.add(state=torch.from_numpy(s),
                   action=torch.from_numpy(baseline_action),
                   reward=torch.from_numpy(np.array([reward * scale_reward])),
                   next_state=torch.from_numpy(s_next),
                   done=done)


def offline_vvc(config):
    scale_reward = config['algo']['scale_reward']
    RL_steps = config['algo']['offline_training_steps']

    replay = ReplayBuffer(replay_size=config['replay_size'],
                          seed=config['seed'])
    env = _env_setup(config)
    env.reset(Mode.OFFLINE)
    _data2replay(env, replay, scale_reward)
    agent = _agent_setup(config, env)

    for iter in tqdm(range(RL_steps), desc="Offline training"):
        agent.update(replay)  # update NN parameters during offline training (while using historical data set)

    offline_res = {'agent': agent,
                   'env': env,
                   'replay': replay}
    return offline_res


def _max_volt_vio(v):
    # performance metric: maximum voltage magnitude violation
    v_max = np.max(v)
    v_min = np.min(v)
    v_vio_max = max(v_max - 1.05 * 120, 0)
    v_vio_min = max(0.95 * 120 - v_min, 0)
    v_vio = max(v_vio_max, v_vio_min)
    return v_vio


def _max_min_volt(v):
    # performance metric: max and min voltage magnitude
    v_max_pu = np.max(v) / 120
    v_min_pu = np.min(v) / 120
    return v_max_pu, v_min_pu


def test_setup(offline_rec):
    # make copy of the whole offline env and reply buffer
    # cant use original objects because of the correlation with online data
    env = copy.deepcopy(offline_rec['env'])
    replay = copy.deepcopy(offline_rec['replay'])

    # set env for testing
    env.reset(Mode.TEST)

    reward_diff = []
    average_reward_diff = []
    v_max_vio = []
    average_v_max_vio = []

    return env, replay, reward_diff, average_reward_diff, v_max_vio, average_v_max_vio


def test_vvc(env, agent, replay, scale_reward):
    reward_diff = []
    v_max_vio = []
    env.reset(Mode.TEST)
    for iter in tqdm(range(env.len_test - 1), desc="Testing"):
        s = env.state
        a = agent.act_deterministic(torch.from_numpy(s)[None, :])
        s_next, reward, done, info = env.step(a)

        replay.add(state=torch.from_numpy(s),
                   action=torch.from_numpy(a),
                   reward=torch.from_numpy(np.array([reward * scale_reward])),
                   next_state=torch.from_numpy(s_next),
                   done=done)

        v_rl = info['v']

        reward_diff.append(reward - info['baseline_reward'])
        v_max_vio.append(_max_volt_vio(v_rl))
    return reward_diff, v_max_vio


def online_vvc(config, offline_rec):
    agent = offline_rec['agent']
    env = offline_rec['env']
    replay = offline_rec['replay']

    scale_reward = config['algo']['scale_reward']

    # number of epochs to be trained
    num_epochs = config['algo']['online_training_steps']

    reward_diff = []
    average_reward_diff = []
    v_max_vio = []
    average_v_max_vio = []

    # setup test environment
    test_env, test_replay, test_reward_diff, test_average_reward_diff, \
    test_v_max_vio, test_average_max_vio = test_setup(offline_rec)

    for epoch in tqdm(range(num_epochs), desc="Online training"):
        env.reset(Mode.ONLINE)
        test_env.reset(Mode.TEST)
        for iter in tqdm(range(env.len_online), desc="Online training_epoch{}".format(epoch)):
            s = env.state
            a = agent.act_probabilistic(torch.from_numpy(s)[None, :])
            s_next, reward, done, info = env.step(a)

            replay.add(state=torch.from_numpy(s),
                       action=torch.from_numpy(a),
                       reward=torch.from_numpy(np.array([reward * scale_reward])),
                       next_state=torch.from_numpy(s_next),
                       done=done)
            agent.update(replay)  # update NN parameters during online training (while interact with the environment)

            v_rl = info['v']

            # train reward and max v
            reward_diff.append(reward - info['baseline_reward'])
            v_max_vio.append(_max_volt_vio(v_rl))

        average_v_max_vio.append(np.average(v_max_vio[-env.len_online + 1]))
        average_reward_diff.append(np.average(reward_diff[-env.len_online + 1]))

        # do test
        epoch_test_reward_diff, epoch_test_v_max_vio = test_vvc(test_env, agent, test_replay, scale_reward)
        # append test results
        test_reward_diff = test_reward_diff + epoch_test_reward_diff
        test_average_reward_diff.append(np.average(epoch_test_reward_diff))
        test_v_max_vio = test_v_max_vio + epoch_test_v_max_vio
        test_average_max_vio.append(np.average(epoch_test_v_max_vio))

    online_res = {'reward_diff (r - rbaseline)': np.array(reward_diff),
                  'average_reward_diff (r - rbaseline)': np.array(average_reward_diff),
                  'max voltage violation': np.array(v_max_vio),
                  'average max voltage violation': np.array(average_v_max_vio),
                  'agent': agent,
                  'env': env}
    test_res = {'reward_diff (r - rbaseline)': np.array(test_reward_diff),
                'average_reward_diff (r - rbaseline)': np.array(test_average_reward_diff),
                'max voltage violation': np.array(test_v_max_vio),
                'average max voltage violation': np.array(test_average_max_vio)}
    return online_res, test_res


def test_vvc_verbose(online_res):
    # make copy of the whole offline env and reply buffer
    # cant use original objects because of the correlation with online data
    env = copy.deepcopy(online_res['env'])
    agent = copy.deepcopy(online_res['agent'])

    # set env for plotting test results
    env.reset(Mode.TEST, start_time=5330)
    len_step = 24  # Because we need the results for 3 consecutive day while our dataset is half hour -> 2*72 =144

    capacitor_status = []
    oltc_position = []
    max_min_voltage = []
    p_load = []
    q_load = []

    for i in range(len_step):
        s = env.state
        a = agent.act_deterministic(torch.from_numpy(s)[None, :])
        s_next, reward, done, info = env.step(a)

        action = a

        voltage = info['v']
        load_kw = info['load_kw']
        load_kvar = info['load_kvar']

        num_of_oltc = len(env.reg_names)
        oltc_position.append(action[:num_of_oltc])
        capacitor_status.append(action[num_of_oltc:])
        max_min_voltage.append(_max_min_volt(voltage))
        p_load.append(load_kw)
        q_load.append(load_kvar)

    test_vvc_res = {'tap position oltc': np.array(oltc_position),
                    'status capacitors': np.array(capacitor_status),
                    'voltage': np.array(max_min_voltage)}
                    #'active power load': np.array(p_load),
                    #'reactive power load': np.array(q_load)}

    print(test_vvc_res)

    return test_vvc_res
