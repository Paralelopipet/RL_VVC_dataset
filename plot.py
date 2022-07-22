import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import numpy as np
import latex

rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 12})
rc('text', usetex=False)


def smooth(y, box_pts):
    # smooth curves by moving average
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def plot_res(envs, algos, metric, smoothing,
             ylabel, xlabel, time_stamps):

    fig, axes = plt.subplots(nrows=len(envs), ncols=2, figsize=(4.*len(envs), 4.*len(envs)))

    for ax, (time_stamp, env) in enumerate(list(zip(time_stamps, envs))):
        res_all = []
        for a in algos:
            with open('./res/data/{}_{}{}.pkl'.format(env, a, time_stamp), 'rb') as f:
                res_all.append(pickle.load(f))

        v_all_online = []
        v_all_test = []
        for res in res_all:
            if res['online_{}'.format(metric)]:
                v_all_online.append(np.array(res['online_{}'.format(metric)]))
            else:
                v_all_online.append([])
            if res['test_{}'.format(metric)]:
                v_all_test.append(np.array(res['test_{}'.format(metric)]))
            else:
                v_all_test.append([])

        for i, v_each in enumerate(v_all_online):
            if isinstance(v_each, np.ndarray):
                axes[ax, 0].plot(smooth(np.percentile(v_each, q=50, axis=0), smoothing), label='online_{}'.format(algos[i]))
                axes[ax, 0].fill_between(x=np.arange(v_each.shape[1] - smoothing + 1),
                                      y1=smooth(np.percentile(v_each, q=10, axis=0), smoothing),
                                      y2=smooth(np.percentile(v_each, q=90, axis=0), smoothing), alpha=0.4)
        # print testing
        for i, v_each in enumerate(v_all_test):
            if isinstance(v_each, np.ndarray):
                axes[ax, 1].plot(smooth(np.percentile(v_each, q=50, axis=0), smoothing), label='test_{}'.format(algos[i]))
                axes[ax, 1].fill_between(x=np.arange(v_each.shape[1] - smoothing + 1),
                                      y1=smooth(np.percentile(v_each, q=10, axis=0), smoothing),
                                      y2=smooth(np.percentile(v_each, q=90, axis=0), smoothing), alpha=0.4)
        axes[ax, 0].grid(True, alpha=0.1)
        axes[ax, 1].grid(True, alpha=0.1)
        if env == '8500':
            axes[ax, 0].title.set_text('{} train node'.format(env))
            axes[ax, 1].title.set_text('{} test node'.format(env))
        else:
            axes[ax, 0].title.set_text('{} train bus'.format(env))
            axes[ax, 1].title.set_text('{} test bus'.format(env))
        if ax == 0:
            axes[ax, 0].set_ylabel(ylabel)
            axes[ax, 1].set_ylabel(ylabel)
        if ax == 1:
            axes[ax, 0].set_xlabel(xlabel)
        axes[ax, 0].legend()
        axes[ax, 1].legend()

    plt.legend()
    plt.savefig('./res/figs/bus{}_{}.pdf'.format(envs, metric), bbox_inches='tight')
    plt.show()
