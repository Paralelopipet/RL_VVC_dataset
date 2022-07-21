import numpy as np
import torch
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from envs.env import VVCEnv
import copy
from enum import Enum
class Mode(Enum):
    OFFLINE = 1
    ONLINE = 2
    TEST = 3

from algos.replay import ReplayBuffer


def _env_setup(config):
    return VVCEnv(config['env'],
                  config['state_option'],
                  config['reward_option'],
                  config['offline_split'],
                  config['online_split'])


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
    RL_steps = config['algo']['training_steps']

    replay = ReplayBuffer(replay_size=config['replay_size'],
                          seed=config['seed'])
    env = _env_setup(config)
    env.reset(Mode.OFFLINE)
    _data2replay(env, replay, scale_reward)
    agent = _agent_setup(config, env)

    for iter in tqdm(range(RL_steps), desc="Offline training"):
        agent.update(replay)

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

def test_vvc(offline_rec):

    # make copy of the whole offline env and reply buffer
    # cant use original objects because of the correlation with online data
    env = copy.deepcopy(offline_rec['env'])
    replay = copy.deepcopy(offline_rec['replay'])

    # set env for testing
    env.reset(Mode.TEST)

    reward_diff = []
    v_max_vio = []

    return env, replay, reward_diff, v_max_vio

def do_test_step(env, agent, replay, scale_reward):
        s = env.state
        a = agent.act_deterministic(torch.from_numpy(s)[None, :])
        s_next, reward, done, info = env.step(a)

        replay.add(state=torch.from_numpy(s),
                   action=torch.from_numpy(a),
                   reward=torch.from_numpy(np.array([reward * scale_reward])),
                   next_state=torch.from_numpy(s_next),
                   done=done)

        v_rl = info['v']

        return reward, v_rl


def online_vvc(config, offline_rec):

    agent = offline_rec['agent']
    env = offline_rec['env']
    replay = offline_rec['replay']

    scale_reward = config['algo']['scale_reward']

    env.reset(Mode.ONLINE)
    reward_diff = []
    v_max_vio = []

    # setup test environment
    test_env, test_replay, test_reward_diff, test_v_max_vio = test_vvc(offline_rec)

    for iter in tqdm(range(env.len_online - 1), desc="Online training"):

        s = env.state
        a = agent.act_probabilistic(torch.from_numpy(s)[None, :])
        s_next, reward, done, info = env.step(a)

        replay.add(state=torch.from_numpy(s),
                   action=torch.from_numpy(a),
                   reward=torch.from_numpy(np.array([reward * scale_reward])),
                   next_state=torch.from_numpy(s_next),
                   done=done)
        agent.update(replay)

        v_rl = info['v']

        #train reward and max v
        reward_diff.append(reward - info['baseline_reward'])
        v_max_vio.append(_max_volt_vio(v_rl))

        # test reward and max v
        test_reward, test_v_rl = do_test_step(test_env, agent, test_replay, scale_reward)
        test_reward_diff.append(test_reward - info['baseline_reward'])
        test_v_max_vio.append(_max_volt_vio(test_v_rl))        

    online_res = {'reward_diff (r - rbaseline)': np.array(reward_diff),
                  'max voltage violation': np.array(v_max_vio)}
    test_res = {'reward_diff (r - rbaseline)': np.array(test_reward_diff),
                  'max voltage violation': np.array(test_v_max_vio)}
    return online_res, test_res
