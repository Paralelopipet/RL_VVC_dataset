from collections import OrderedDict
# An OrderedDict is a dictionary subclass that remembers the order in which its contents are added
# since regular dictionary doest not recall data in right order.

"""This file includes load information, load name, load phase, kW and kVar, from different DNs, 13, 123, 8500"""


def load_info(env_name):
    # input: env_name: 13, 123, 8500
    # output:
    #   1. load_base: OrderedDict(key=load name, value=kw, kvar of the IEEE feeders snapshot)
    #      for 8500, value=kw only
    #   2. load_node: OrderedDict(key=bus name of load, value=phases of load)

    if str(env_name) == '13':  # str() coverts a specified value to a string
        # load_base: key=name of load object in the dss script
        #            value=(kw, k-var)
        load_base = OrderedDict(
            [('671', (1155, 660)),  # load name: (kw, kvar)
             ('634a', (160, 110)),
             ('634b', (120, 90)),
             ('634c', (120, 90)),
             ('645', (170, 125)),
             ('646', (230, 132)),
             ('692', (170, 151)),
             ('675a', (485, 190)),
             ('675b', (68, 60)),
             ('675c', (290, 212)),
             ('611', (170, 80)),
             ('652', (128, 86)),
             ('670a', (17, 10)),
             ('670b', (66, 38)),
             ('670c', (117, 68)),
             ])

        # load_node: key=name of load object in the dss script
        #            value= phases of the load. Must be in the order (a,b,c)
        load_node = OrderedDict(
            [('671', ''),
             ('634', 'abc'),
             ('645', ''),
             ('646', ''),
             ('692', ''),
             ('675', 'abc'),
             ('611', ''),
             ('652', ''),
             ('670', 'abc'),
            ])

    elif str(env_name) == '123':
        with open("./envs/dss_123/IEEE123Loads.DSS", 'r') as dssfile:
            dss_str = dssfile.readlines()

        load_base = OrderedDict()
        for s in dss_str:
            if 'New Load.' in s:
                idx = s.index("New Load.") + len('New Load.')  # s.index("New Load.") = 0 len('New Load.') = 9
                name = []
                for c in s[idx:]:
                    if c == ' ':
                        break
                    else:
                        name.append(c)
                name = ''.join(name)  # join all items in a tuple into a string and use '' as separator.

                idx_kW = s.index("kW=") + len('kW=')
                kW = []
                for c in s[idx_kW:]:
                    if c == ' ':
                        break
                    else:
                        kW.append(c)
                kW = float(''.join(kW))

                idx_kvar = s.lower().index("kvar=") + len('kvar=')
                kvar = []
                for c in s[idx_kvar:]:
                    if c == ' ':
                        break
                    else:
                        kvar.append(c)
                kvar = float(''.join(kvar))

                load_base[name] = (kW, kvar)

        load_node = OrderedDict()
        phases = {'1': 'a',
                  '2': 'b',
                  '3': 'c',
                  ' ': ''}
        for s in dss_str:
            if 'Bus1=' in s:
                idx = s.index("Bus1=") + len('Bus1=')
                name = []
                for c in s[idx:]:
                    if c == '.':
                        p = s[idx + len(name) + 1]
                        break
                    elif c == ' ':
                        p = ' '
                        break
                    else:
                        name.append(c)

                name = ''.join(name).lower()

                load_node[name] = phases[p]

    elif str(env_name) == '8500':
        with open("./envs/dss_8500/Loads.dss", 'r') as dssfile:
            dss_str = dssfile.readlines()

        load_base = OrderedDict()
        for s in dss_str:
            if 'New Load.' in s:
                idx_name = s.index("New Load.") + len('New Load.')
                name = []
                for c in s[idx_name:]:
                    if c == ' ':
                        break
                    else:
                        name.append(c)
                name = ''.join(name)

                idx_kW = s.index("kW=") + len('kW=')
                kW = []
                for c in s[idx_kW:]:
                    if c == ' ':
                        break
                    else:
                        kW.append(c)
                kW = float(''.join(kW))

                load_base[name] = kW

        load_node = OrderedDict()
        for s in dss_str:
            if 'Bus1=' in s:
                idx_name = s.index("Bus1=") + len('Bus1=')
                name = []
                for c in s[idx_name:]:
                    if c == '.':
                        break
                    else:
                        name.append(c)
                name = ''.join(name).lower()

                load_node[name] = ''

    return load_base, load_node






import numpy as np
from numpy import genfromtxt
# The genfromtxt() used to load data from a text file, with missing values handled as specified.
import cmath
# Use for working with complex numbers 1+2j
import opendssdirect as dss
from collections import defaultdict
# An OrderedDict is a dictionary subclass that remembers the order in which its contents are added
# since regular dictionary doest not recall data in right order
#from .env_utils import load_info


class VVCEnv:
    def __init__(self,
                 env='13',
                 state_option=1,
                 reward_option=1,
                 offline_split=0.7,
                 online_split=0.3):
        self.env_name = str(env)
        self.state_option = str(state_option)
        self.reward_option = str(reward_option)
        assert offline_split > 0.0 and offline_split < 1.0, "'offline_split' must be in (0.0, 1.0)"
        assert online_split > 0.0 and online_split < 1.0, "'online_split' must be in (0.0, 1.0)"
        sum_split = offline_split + online_split
        assert sum_split > 0.0 and sum_split <= 1.0, "'offline_split' + 'online_split' must be in (0.0, 1.0]"
        self.offline_split = offline_split
        self.online_split = online_split
        # The assert keyword lets you test if a condition in your code returns True, if not, the program will raise an AssertionError.

        # offline data & online data size
        self.len_offline = round(27649 * self.offline_split)  # 19354
        self.len_online = round(27649 * self.online_split)  # 8295
        self.len_total = self.len_offline + self.len_online
        #print(self.len_total)

        # circuit model & raw ami data
        if self.env_name == '13':
            dss.run_command('Redirect envs/dss_13/IEEE13Nodeckt.dss')
            self.basekVA = 5000.
            self.ami_data = genfromtxt('./data/processed/first_2897_ami_aggto_580.csv', delimiter=',', max_rows=self.len_total)
            self.ami_data = self.ami_data / (np.mean(self.ami_data, axis=0) * 2.0)
            # the 13 bus feeder is highly loaded, an extra 2.0 is divided
        elif self.env_name == '123':
            dss.run_command('Redirect envs/dss_123/IEEE123Master.dss')
            self.basekVA = 5000.
            self.ami_data = genfromtxt('./data/processed/first_2897_ami_aggto_580.csv', delimiter=',', max_rows=self.len_total)
            self.ami_data = self.ami_data / (np.mean(self.ami_data, axis=0) * 1.0)
        elif self.env_name == '8500':
            dss.run_command('Redirect envs/dss_8500/Master.dss')
            self.basekVA = 27.5 * 1000
            self.ami_data = genfromtxt('./data/processed/first_2897_ami.csv', delimiter=',', max_rows=self.len_total)
            self.ami_data = self.ami_data / (np.mean(self.ami_data, axis=0) * 2.0)

        # load info to get nominal values for kW and kVar
        self.load_base, self.load_node = load_info(self.env_name)

        # vvc devices
        self.reg_names = dss.RegControls.AllNames()
        self.cap_names = dss.Capacitors.AllNames()

        # vvc data (offline & online)
        self.loss = genfromtxt('./data/processed/{}/loss.csv'.format(self.env_name), delimiter=',', max_rows=self.len_total)[:, None]
        self.substation_pq = genfromtxt('./data/processed/{}/substation_pq.csv'.format(self.env_name), delimiter=',', max_rows=self.len_total)
        self.load = genfromtxt('./data/processed/{}/load.csv'.format(self.env_name), delimiter=',', max_rows=self.len_total)
        self.volt = genfromtxt('./data/processed/{}/volt.csv'.format(self.env_name), delimiter=',', max_rows=self.len_total)
        self.ltc_tap = genfromtxt('./data/processed/{}/tap.csv'.format(self.env_name), delimiter=',', max_rows=self.len_total)
        if len(self.ltc_tap.shape) == 1:
            self.ltc_tap = self.ltc_tap[:, None]
            # I do not know why this if is written
        self.cap_status = genfromtxt('./data/processed/{}/status.csv'.format(self.env_name), delimiter=',', max_rows=self.len_total)

        self.load_avg = np.average(self.load[:self.len_offline, :], axis=0)

        # RL info
        self.dims_time = (168 * 2, 24 * 2)  # period of time. e.g. for weekly pattern use 168 * 2 for half-hourly data
        self.dims_ltc = (33, ) * len(self.reg_names)  # 33 tap position for OLTC
        self.dims_cap = (2, ) * len(self.cap_names)  # on or off = 2
        self.dims_action = self.dims_ltc + self.dims_cap  # dimention of action
        self.dim_substation_pq = self.substation_pq.shape[1]
        self.dim_load = self.load.shape[1]
        self.dim_volt = self.volt.shape[1]  # take the second value as dimention for the bus voltages at each feeder

        #reward coefficients
        self.coef_switching = 0.1
        self.coef_volt = 0.5
        if self.reward_option in ('1', '3'):
            self.coef_loss = 1.0
        elif self.reward_option in ('2', '4'):
            self.coef_loss = 0.0

        if self.state_option in ('1', '3'):
            self.dim_state = self.dim_substation_pq + self.dim_load + len(self.dims_action) + 2*len(self.dims_time)
            # the 2 in the last term is for cos AND sin encoding of periodic variable
        elif self.state_option in ('2', ):
            self.dim_state = self.dim_substation_pq + len(self.dims_action) + 2*len(self.dims_time)

        self.global_time = None
        self.state = None
        self.action_prev = None

    def reset(self, offline=True):
        # if offline = True, set the global time to the beginning of the full dataset;
        # otherwise set the global time to the beginning of the online dataset
        if offline:
            self.global_time = 1
        else:
            self.global_time = max(self.len_offline, 1)

        if self.state_option == '3':
            self.global_time = max(self.global_time, 48)
            # to consider communication delay

        if self.state_option == '1':
            self.state = np.concatenate([
                self.substation_pq[self.global_time-1, :] / (self.basekVA / 3.),
                self.load[self.global_time, :] / self.load_avg,
            ]) # joint pq and load as state in the same time
            # Concatenation refers to joining. This function is used to join two or more arrays of the same shape along
            # a specified axis.
        elif self.state_option == '2':
            self.state = np.concatenate([
                self.substation_pq[self.global_time-1, :] / (self.basekVA / 3.),
            ])
        elif self.state_option == '3':
            self.state = np.concatenate([
                self.substation_pq[self.global_time-1, :] / (self.basekVA / 3.),
                self.load[self.global_time - 48, :] / self.load_avg,
            ])
        self.state = np.concatenate([self.state,
                                     self.ltc_tap[self.global_time-1, :],
                                     self.cap_status[self.global_time-1, :],
                                     np.array([np.cos(2 * np.pi * (self.global_time / ii)) for ii in self.dims_time]),
                                     np.array([np.sin(2 * np.pi * (self.global_time / ii)) for ii in self.dims_time])
                                     ])
        self.action_prev = np.concatenate((self.ltc_tap[self.global_time-1, :],
                                           self.cap_status[self.global_time-1, :]))
        # joint CBs status and OLTC tap changing as action!
        return self.state

    @staticmethod
    def tap_to_tappu(tap):
        # from [0, 32] to [0.9, 1.1]
        pu_per_ltc_tap = 5/8/100  # 5/8 % voltage rule
        tap_pu = 1.0 + (tap - 16) * pu_per_ltc_tap
        return tap_pu

    @staticmethod
    def tappu_to_tap(tap_pu):
        # from [0.9, 1.1] to [0, 32]
        pu_per_ltc_tap = 5/8/100
        tap = (tap_pu - 1.0) / pu_per_ltc_tap + 16
        return tap

    @staticmethod
    def average_every_n(nparr, n):
        # given 1-D np array and an int n, return array([np.average(nparr[:n]), np.average(nparr[n:2*n]), ...])
        res = []
        for i in range(len(nparr) // n):
            res.append(np.average(nparr[i * n:(i + 1) * n]))
        res.append(np.average(nparr[len(nparr) // n * n:]))
        return np.array(res)

    def step(self, action=None):
        info = defaultdict(lambda: None)

        if action is None:
            # if no action is provided, use the dss default control logic and call historical data
            action = np.concatenate([self.ltc_tap[self.global_time, :], self.cap_status[self.global_time, :]])
            # joint CBs status and OLTC tap ratio as action
            volt_120 = self.volt[self.global_time, :]
            volt_pu = volt_120 / 120.0
            loss = self.loss[self.global_time, 0]
            substation_pq = self.substation_pq[self.global_time, :]
            info['PF converge'] = True
        else:
            # set load
            if self.env_name == '8500':
                load_kw = [val * self.ami_data[self.global_time, i] for i, (key, val) in enumerate(self.load_base.items())]
                for i, (key, val) in enumerate(self.load_base.items()):
                    dss.run_command('Load.{}.kW={}'.format(key, load_kw[i]))
            elif self.env_name in ('13', '123'):
                load_kw = [val[0] * self.ami_data[self.global_time, i] for i, (key, val) in enumerate(self.load_base.items())]
                load_kvar = [val[1] * self.ami_data[self.global_time, i] for i, (key, val) in enumerate(self.load_base.items())]
                for i, (key, val) in enumerate(self.load_base.items()):
                    dss.run_command('Load.{}.kW={}'.format(key, load_kw[i]))
                for i, (key, val) in enumerate(self.load_base.items()):
                    dss.run_command('Load.{}.kvar={}'.format(key, load_kvar[i]))

            # set ltc tap and cap status (OpenDSS manual. pp-117)
            for i, reg in enumerate(self.reg_names):
                tap = self.tap_to_tappu(action[i])
                dss.run_command('transformer.{}.Taps=[1.0 {}]'.format(reg, tap))
            for i, cap in enumerate(self.cap_names):
                dss.run_command('Capacitor.{}.status={}'.format(cap, action[i+len(self.reg_names)]))

            # solve
            dss.run_command('Set Controlmode=OFF') # the power flow is not static any more
            dss.Solution.Solve()

            if not dss.Solution.Converged():
                info['PF converge'] = False
            else:
                # voltage profile
                # ref.1: https://sourceforge.net/p/electricdss/discussion/861977/thread/a53badb5/
                # ref.2: https://github.com/dss-extensions/OpenDSSDirect.py/issues/15
                if self.env_name == '8500':
                    volt = []
                    for bus in self.load_node.keys():
                        dss.Circuit.SetActiveBus(bus)
                        v_mag_angle = dss.Bus.puVmagAngle()
                        v = v_mag_angle[::2]
                        ang = v_mag_angle[1::2]
                        v1 = cmath.rect(v[0], ang[0] / 180. * np.pi)
                        v2 = cmath.rect(v[1], ang[1] / 180. * np.pi)
                        v0 = abs(v1 - v2) / 2.0
                        volt.append(v0)  # smart meter measures phase-phase volt
                    volt_120 = np.around(np.array(volt) * 120.0, decimals=1)
                    volt_120 = self.average_every_n(volt_120, 10)
                    volt_pu = volt_120 / 120.0
                elif self.env_name in ('13', '123'):
                    volt = []
                    for bus in self.load_node.keys():
                        dss.Circuit.SetActiveBus(bus)
                        v_mag_angle = dss.Bus.puVmagAngle()
                        v = v_mag_angle[::2]
                        if self.load_node[bus]:
                            for i in range(len(self.load_node[bus])):
                                volt.append(v[i])
                        else:
                            volt.append(v[0])
                    volt_120 = np.around(np.array(volt) * 120.0, decimals=1)
                    volt_pu = volt_120 / 120.0

                # loss (kw)
                loss = np.around(np.array(dss.Circuit.Losses()[0] / 1000.), decimals=1)

                # total power (kw, kvar)
                # ref.1: https://sourceforge.net/p/electricdss/discussion/beginners/thread/6d771703/#0344
                dss.Circuit.SetActiveElement('Vsource.source')
                substation_pq = dss.CktElement.Powers()[:6]
                substation_p = substation_pq[::2]
                substation_q = substation_pq[1::2]
                substation_pq = np.around(-np.array(substation_p + substation_q), decimals=1)

                info['PF converge'] = True

        # reward
        if info['PF converge']:
            if self.reward_option in ('1', '2'):
                reward = - (np.sum(np.round(np.abs(action - self.action_prev))) * self.coef_switching +
                            np.sum(np.abs(volt_pu - 1.0)) * self.coef_volt +
                            loss / self.basekVA * self.coef_loss)
            elif self.reward_option in ('3', '4'):
                reward = - (np.sum(np.round(np.abs(action - self.action_prev))) * self.coef_switching +
                            np.sum(np.logical_or(volt_pu < 0.95, volt_pu > 1.05).astype(float)) * self.coef_volt +
                            loss / self.basekVA * self.coef_loss)
        else:
            if self.reward_option in ('1', '2'):
                reward = - (np.sum(np.round(np.abs(action - self.action_prev))) * self.coef_switching +
                            0.5 * self.dim_volt * self.coef_volt +
                            1.0 * self.coef_loss) * 10.
            elif self.reward_option in ('3', '4'):
                reward = - (np.sum(np.round(np.abs(action - self.action_prev))) * self.coef_switching +
                            1.0 * self.dim_volt * self.coef_volt +
                            1.0 * self.coef_loss) * 10.

        # baseline reward (reward under dss policy)
        baseline_action = np.concatenate([self.ltc_tap[self.global_time, :], self.cap_status[self.global_time, :]])
        baseline_volt_120 = self.volt[self.global_time, :]
        baseline_volt_pu = baseline_volt_120 / 120.0
        baseline_loss = self.loss[self.global_time, 0]
        if self.reward_option in ('1', '2'):
            info['baseline_reward'] = - (np.sum(np.round(np.abs(baseline_action - self.action_prev))) * self.coef_switching +
                                         np.sum(np.abs(baseline_volt_pu - 1.0)) * self.coef_volt +
                                         baseline_loss / self.basekVA * self.coef_loss)
        elif self.reward_option in ('3', '4'):
            info['baseline_reward'] = - (np.sum(np.round(np.abs(baseline_action - self.action_prev))) * self.coef_switching +
                                         np.sum(np.logical_or(baseline_volt_pu < 0.95, volt_pu > 1.05).astype(float)) * self.coef_volt +
                                         baseline_loss / self.basekVA * self.coef_loss)

        # next state
        self.global_time += 1

        if not info['PF converge']:
            substation_pq = self.substation_pq[self.global_time-1, :]
        if self.state_option == '1':
            self.state = np.concatenate([
                substation_pq / (self.basekVA / 3.),
                self.load[self.global_time, :] / self.load_avg,
            ])
        elif self.state_option == '2':
            self.state = np.concatenate([
                substation_pq / (self.basekVA / 3.),
            ])
        elif self.state_option == '3':
            self.state = np.concatenate([
                substation_pq / (self.basekVA / 3.),
                self.load[self.global_time-48, :] / self.load_avg,
            ])
        self.state = np.concatenate([self.state,
                                     action,
                                     np.array([np.cos(2 * np.pi * (self.global_time / ii)) for ii in self.dims_time]),
                                     np.array([np.sin(2 * np.pi * (self.global_time / ii)) for ii in self.dims_time])
                                     ])

        if info['PF converge']:
            info['loss'] = loss
            info['v'] = volt_120
            info['substation_pq'] = substation_pq
        info['switching_steps'] = np.sum(np.round(np.abs(action - self.action_prev)))
        info['action'] = action

        self.action_prev = action.copy() # what does it mean? previous

        if self.global_time == self.len_total:
            done = True
            self.global_time = None
        else:
            done = False

        return self.state, reward, done, info

    # def step {action, reward, next_state, info}

