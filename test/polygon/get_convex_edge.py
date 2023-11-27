#!/usr/bin/env python3

import matplotlib.pyplot as plt

from pathpatrol.polygon import Polygon

p_gon = Polygon([
    [-3.0, 1.0], # 0
    [-1.0, 0.0], # 1
	[0.0, 0.0], # 2
	[1.0, -1.0], # 3
	[2.0, 2.0], # 4
	[3.0, -2.0], # 5
	[5.0, 1.0], # 6
	[7.0, 0.0], # 7
    [11.0, 0.0] # 8
])

plt.figure()
p_gon.plot()
r_lst = p_gon.get_convex_edge(2, 7)
plt.plot([p_gon.x_arr[r] for r in r_lst], [p_gon.y_arr[r] for r in r_lst])
r_lst = p_gon.get_convex_edge(7, 2)
plt.plot([p_gon.x_arr[r] for r in r_lst], [p_gon.y_arr[r] for r in r_lst])
plt.grid()
plt.axis("equal")
plt.show()
