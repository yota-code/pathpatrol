#!/usr/bin/env python3

""" sans optimisation, tout polygon est utilisé complètement ! """

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon
from pathpatrol.compute import Compute
from pathpatrol.sequence import Point
from pathpatrol.route import Route

layer = Layer().load_rastermap('map.png')

A = Point(40.0, 40.0)
B = Point(250.0, 240.0)

layer.g_lst = [layer.g_lst[0],]

u = Compute(layer).solve(A, B)

# P, Q = Vertex(u.layer, 0, 485), Vertex(u.layer, 0, 206)
# u.route.push(P, Q, None)
# u.goaround_corner(P, Q)

# """ il faut rechercher les polygones qui intersectent avec AB et les prendre dans l'ordre, de A vers B"""



# plt.figure(figsize=(16, 16))
# layer.plot()
# plt.plot([A[0], B[0]], [A[1], B[1]], '+-', color="tab:orange")
# plt.grid()
# plt.axis('equal')
# plt.savefig("00.png")
# plt.close()