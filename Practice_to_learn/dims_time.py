import math

import numpy as np
from numpy import genfromtxt
import cmath
"""
dims_time = (168 * 2, 24 * 2)
global_time = 1
a = 2 * len(dims_time)
print(a)
"""
for i in range(1, 24):
    print(i)
    print(math.cos((math.pi * i) / 24))
    print(math.sin((math.pi * i)/24))

# t = np.concatenate([np.array([np.cos(2 * np.pi * (global_time / ii)) for ii in dims_time]),
                    #np.array([np.sin(2 * np.pi * (global_time / ii)) for ii in dims_time])])
