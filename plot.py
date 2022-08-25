import matplotlib.pyplot as plt
from matplotlib import rc
import pickle
import numpy as np

rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 12})
rc('text', usetex=False)


def smooth(y, box_pts):
    # smooth curves by moving average
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def plot_res1(envs, algos, metric, smoothing, ylabel, xlabel, time_stamp):
    fig, axes = plt.subplots(nrows=max(len(envs),2), ncols=2, figsize=(4. * len(envs), 4. * len(envs)))

    for ax, env in enumerate(envs):
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
                axes[ax, 0].plot(smooth(np.percentile(v_each, q=50, axis=0), smoothing),
                                 label='online_{}'.format(algos[i]))
                axes[ax, 0].fill_between(x=np.arange(v_each.shape[1] - smoothing + 1),
                                         y1=smooth(np.percentile(v_each, q=10, axis=0), smoothing),
                                         y2=smooth(np.percentile(v_each, q=90, axis=0), smoothing), alpha=0.4)
        # print testing
        for i, v_each in enumerate(v_all_test):
            if isinstance(v_each, np.ndarray):
                axes[ax, 1].plot(smooth(np.percentile(v_each, q=50, axis=0), smoothing),
                                 label='test_{}'.format(algos[i]))
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
            #axes[ax, 0].set_xlabel(xlabel)
            axes[ax, 1].set_ylabel(ylabel)
            #axes[ax, 1].set_xlabel(xlabel)
        if ax == 1:
            axes[ax, 0].set_ylabel(ylabel)
            axes[ax, 0].set_xlabel(xlabel)
            axes[ax, 1].set_ylabel(ylabel)
            axes[ax, 1].set_xlabel(xlabel)

        axes[ax, 0].legend()
        axes[ax, 1].legend()

    plt.legend()  # label
    plt.savefig('./res/figs/bus{}_{}.pdf'.format(envs, metric), bbox_inches='tight')
    plt.show()


def plot_res2(test_vvc_res, env, algos):
    metric = ['tap position oltc', 'status capacitors', 'voltage', 'load feeder']
    fig, axes = plt.subplots(nrows=len(metric), ncols=1, figsize=(4. * len(metric), 4. * len(metric)))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    if env == '13':
        #print("13")
        voltages = test_vvc_res['voltage']
        maxVoltages = voltages[:, 0]
        minVoltages = voltages[:, 1]
        axes[0].plot(minVoltages, label='Min')
        axes[0].plot(maxVoltages, label='Max')

        capacitorStatuses = test_vvc_res['status capacitors']
        axes[1].plot(capacitorStatuses, label=['CB1', 'CB2'])

        tapPositions = test_vvc_res['tap position oltc']
        axes[2].plot(tapPositions, label='OLTC1')

        activePowerFeeder = test_vvc_res['active power feeder']
        reactivePowerFeeder = test_vvc_res['reactive power feeder']
        x = np.arange(test_vvc_res['len_step'])
        axes[3].bar(x, height=activePowerFeeder, width=0.9, align='center', label='active power load')
        axes[3].bar(x, height=reactivePowerFeeder, width=0.9, align='center', label='reactive power load')

        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[3].legend()

        axes[0].set_ylabel('voltage p.u.')
        axes[1].set_ylabel('CB status')
        axes[2].set_ylabel('OLTC tap position')
        axes[3].set_ylabel('load (kW/kVar)')
        axes[3].set_xlabel('Time (half-hour)')

        axes[0].grid(True, axis='x', alpha=0.5)
        axes[1].grid(True, axis='x', alpha=0.5)
        axes[2].grid(True, axis='x', alpha=0.5)
        axes[3].grid(True, axis='x', alpha=0.5)

        axes[0].title.set_text('{} test bus'.format(env))

        # Make data for 3D plot
        X = np.arange(1, 16, 1)
        Y = np.arange(0, test_vvc_res['len_step'], 1)
        X, Y = np.meshgrid(X, Y)
        Z = test_vvc_res['voltage all buses']

        ax.view_init(10, 45)

        ax.set_xlabel('Bus')
        ax.set_ylabel('Time (half-hour)')
        ax.set_zlabel('Voltage (p.u.)')

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.xaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

    if env == '123':
        #print("123")
        voltages = test_vvc_res['voltage']
        maxVoltages = voltages[:, 0]
        minVoltages = voltages[:, 1]
        axes[0].plot(minVoltages, label='Min')
        axes[0].plot(maxVoltages, label='Max')

        capacitorStatuses = test_vvc_res['status capacitors']
        axes[1].plot(capacitorStatuses, label=['CB1', 'CB2', 'CB3', 'CB4'])

        tapPositions = test_vvc_res['tap position oltc']
        axes[2].plot(tapPositions, label=['OLTC1', 'OLTC2', 'OLTC3', 'OLTC4', 'OLTC5'])

        activePowerFeeder = test_vvc_res['active power feeder']
        reactivePowerFeeder = test_vvc_res['reactive power feeder']
        x = np.arange(test_vvc_res['len_step'])
        axes[3].bar(x, height=activePowerFeeder, width=0.9, align='center', label='active power load')
        axes[3].bar(x, height=reactivePowerFeeder, width=0.9, align='center', label='reactive power load')

        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        axes[3].legend()

        axes[0].set_ylabel('voltage p.u.')
        axes[1].set_ylabel('CB status')
        axes[2].set_ylabel('OLTC tap position')
        axes[3].set_ylabel('load (kW/kVar)')
        axes[3].set_xlabel('Time (half-hour)')

        axes[0].grid(True, axis='x', alpha=0.5)
        axes[1].grid(True, axis='x', alpha=0.5)
        axes[2].grid(True, axis='x', alpha=0.5)
        axes[3].grid(True, axis='x', alpha=0.5)

        axes[0].title.set_text('{} test bus'.format(env))

        # Make data for 3D plot
        X = np.arange(1, 86, 1)
        Y = np.arange(0, test_vvc_res['len_step'], 1)
        X, Y = np.meshgrid(X, Y)
        Z = test_vvc_res['voltage all buses']

        ax.view_init(10, 45)

        ax.set_xlabel('Bus')
        ax.set_ylabel('Time (half-hour)')
        ax.set_zlabel('Voltage (p.u.)')

        # Get rid of colored axes planes
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.xaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})
        ax.yaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})
        ax.zaxis._axinfo["grid"].update({"linewidth": 0.1, "color": "gray"})

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.legend()
    plt.savefig('./res/figs/bus{}_VVC_Result.pdf'.format(env), bbox_inches='tight')
    plt.show()
