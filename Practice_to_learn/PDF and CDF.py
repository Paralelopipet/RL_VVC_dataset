import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

x = np.linspace(0.3, 1.75, 1000)
plt.plot(x, stats.norm.logpdf(x, 1.075, 0.2))
plt.plot(x, stats.norm.pdf(x, 1.075, 0.2))

plt.show()