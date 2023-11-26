#!/usr/bin/env python3

""" sans optimisation, tout polygon est utilisé complètement ! """

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon
from pathpatrol.route import Route, Point, Vertex

layer = Layer().load_rastermap('map.png')

layer.g_lst = [layer.g_lst[0],]

A = (40.0, 40.0)
B = (250.0, 240.0)



u = Route(layer)

u.goaround_corner(Vertex(u.layer, 0, 485), Vertex(u.layer, 0, 206))

# u.compute(A, B)



# """ il faut rechercher les polygones qui intersectent avec AB et les prendre dans l'ordre, de A vers B"""



# plt.figure(figsize=(16, 16))
# layer.plot()
# plt.plot([A[0], B[0]], [A[1], B[1]], '+-', color="tab:orange")
# plt.grid()
# plt.axis('equal')
# plt.savefig("00.png")
# plt.close()