#!/usr/bin/env python3

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.polygon import Polygon

t_arr = np.arange(14) * math.tau / 14
x_arr = 3.0 * (1 - np.cos(t_arr)) * np.cos(t_arr)
y_arr = 3.0 * (1 - np.cos(t_arr)) * np.sin(t_arr)

print(t_arr)
print(x_arr)
print(y_arr)

u = Polygon(x_lst=x_arr, y_lst=y_arr)

M = (-1, -6)
A = math.atan2(u.y_arr[0] - M[1], u.x_arr[0] - M[0])
m_arr = u.ventilate(M) + A


plt.figure()
plt.subplot(2,1,1)
u.plot()
for m in m_arr :
	plt.plot([M[0], M[0] + 10.0 * math.cos(m)], [M[1], M[1] + 10.0 * math.sin(m)], '--', color='pink')
plt.grid()
plt.subplot(2,1,2)
plt.plot(m_arr)
plt.grid()
plt.show()