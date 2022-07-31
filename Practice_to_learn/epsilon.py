import numpy as np
from numpy import random
import matplotlib.pyplot as plt

explore = np.random.binomial(1, 0.02)
print(explore)

plt.plot(random.binomial(1, p=0.02, size=1000), label='binomial')

plt.show()
