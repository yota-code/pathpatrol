#!/usr/bin/env python3

""" sans optimisation, tout polygon est utilisé complètement ! """

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon

layer = Layer().load_rastermap('map.png')

A = (40, 40)
B = (250, 240)

p_map = {
	A : None,
	B : None,
}

p_set = set()
for p in p_map :
	if p_map[p] is None :
		p_map[p] = dict()
		for i, piece in enumerate(layer) :
			p_map[p][i] = piece.get_point_type(A)
			p_set.append[i]

""" il faut rechercher les polygones qui intersectent avec AB et les prendre dans l'ordre, de A vers B"""



plt.figure(figsize=(16, 16))
layer.plot()
plt.plot([A[0], B[0]], [A[1], B[1]], '+-', color="tab:orange")
plt.grid()
plt.axis('equal')
plt.savefig("00.png")
plt.close()