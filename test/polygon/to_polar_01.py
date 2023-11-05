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

A = (-1, -6)
B = (1, 6)
m_arr, d_arr = u.to_polar(A, B)
print("m_arr:", m_arr)

plt.figure()
plt.subplot(2,1,1)
u.plot()
for m, d in zip(m_arr, d_arr) :
	plt.plot([A[0], A[0] + d * math.cos(m)], [A[1], A[1] + d * math.sin(m)], '--', color='pink')
plt.grid()
plt.subplot(2,1,2)
plt.plot(m_arr)
plt.grid()
plt.show()