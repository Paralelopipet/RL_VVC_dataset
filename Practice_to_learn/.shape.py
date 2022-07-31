from numpy import genfromtxt

load = genfromtxt('./data/processed/{}/load.csv'.format(env_name), delimiter=',', max_rows=27649)
