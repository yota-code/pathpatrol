#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

class Layer() :
	def __init__(self) :
		pass
	
	def load_rastermap(self, img_pth) :
		from pathpatrol.rastermap import RasterMap

		lvl = plt.imread('map.png')[:,:,0].astype(np.uint16)
		r_map = RasterMap(lvl)

		self.g_lst = list(r_map.segment())

	def compute_route(self) :
		