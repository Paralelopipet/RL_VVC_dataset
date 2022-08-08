
from plot import plot_res1
plot_res1(envs=['13', '123', '8500'],
         algos=['dqn', 'sac'],
         metric='average max voltage violation',
         smoothing=1,
         ylabel='average max voltage violation',
         xlabel='Timestamp (half-hour)',
         time_stamp = '08_08_2022_18_55_47')