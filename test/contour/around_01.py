#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pawpatrol.contour import Contour_Cartesian

u = Contour_Cartesian()
u.load_json(Path('../demo_01/contour_8.json'))

A, B = (10, 200), (150, 220)

left, right = u.around(A, B)

plt.plot(u.x_lst, u.y_lst)
plt.plot([A[0], B[0]], [A[1], B[1]])
plt.plot([x for x, y in left], [y for x, y in left], linewidth=6, alpha=0.5)
plt.plot([x for x, y in right], [y for x, y in right], linewidth=6, alpha=0.5)

plt.grid()
plt.axis('equal')

plt.show()

