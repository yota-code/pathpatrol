#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer

u = Layer()
u.load_rastermap('map.png')
u.compute_route()
u.plot()

# lvl = plt.imread('map.png')[:,:,0].astype(np.uint16)

# print(u.g_lst)

# u = RasterMap(lvl)

# Path("contour.pson").write_text(repr(u.g_map))

# # plt.imshow(u.lvl, origin='lower')
# # for n in u.g_map :
# # 	plt.plot(u.g_map[n].x_arr, u.g_map[n].y_arr, '+--', label=f"{n}")

# # plt.legend()
# # plt.show()

# # pathpatrol.bridge.find_crosslines(u.g_map[4], u.g_map[3])
# pathpatrol.bridge.find_outlines(u.g_map[4], u.g_map[3])

