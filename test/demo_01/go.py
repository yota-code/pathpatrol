#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.rastermap import RasterMap
from pathpatrol.vectormap import VectorMap

lvl = plt.imread('map.png')[:,:,0].astype(np.uint16)

u = RasterMap(lvl)

v = VectorMap(u.g_map)

plt.imshow(u.lvl, origin='lower')
for n in v.h_map :
	plt.plot(v.h_map[n].x_lst, v.h_map[n].y_lst, '+--', label=f"{n}")
	v.h_map[n].save_json(Path(f"contour_{n}.json"))

plt.legend()
plt.show()



