# -*- coding: utf-8 -*-
"""

Sigmoid函数绘制
@author: Jerry
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,100)
y = 1.0/(1+np.exp(-x))

fig = plt.figure()
plt.plot(x,y)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
