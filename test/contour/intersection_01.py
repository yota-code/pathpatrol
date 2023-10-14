#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pawpatrol.contour import Contour_Cartesian


test_lst = [
	# [(1, 1), (3, 2), (2, -4), (1, 6)],
	# [(1, 0), (4, 0), (3, 0), (5, 0)],
	# [(0.0, 0.5), (2.1, 0.5), (0.1, 0), (1.1, 0)],
	# [(0.0, 0.5), (2.1, 0.5), (1.1, 0), (0.9, 1)],
	# [(0.0, 0.5), (2.1, 0.5), (0.9, 1), (-0.1, 1)],
	# [(0.0, 0.5), (2.1, 0.5), (-0.1, 1), (0.1, 0)],
	[(1.0, 0.5), (2.1, 0.5), (0.1, 0), (1.1, 0)],
	[(1.0, 0.5), (2.1, 0.5), (1.1, 0), (0.9, 1)],
	[(1.0, 0.5), (2.1, 0.5), (0.9, 1), (-0.1, 1)],
	[(1.0, 0.5), (2.1, 0.5), (-0.1, 1), (0.1, 0)],
]

u = Contour_Cartesian()

for A, B, C, D in test_lst :
	k1, k2 = u.intersection(A, B, C, D)

	plt.plot([A[0], B[0]], [A[1], B[1]], 'x-')
	plt.plot([C[0], D[0]], [C[1], D[1]], 'x-')
	plt.plot([A[0] * (1 - k1) + B[0] * k1,], [A[1] * (1 - k1) + B[1] * k1,], 'o')
	plt.plot([C[0] * (1 - k2) + D[0] * k2,], [C[1] * (1 - k2) + D[1] * k2,], '+')
	plt.title(f"k1={k1:.5f}, k2={k2:.5f}")
	plt.grid()
	plt.axis("equal")
	plt.show()

