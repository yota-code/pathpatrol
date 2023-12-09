#!/usr/bin/env python3

""" sans optimisation, tout polygon est utilisé complètement ! """

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon
from pathpatrol.compute import Compute
from pathpatrol.sequence import Point, Vertex
from pathpatrol.route import Route

layer = Layer().load_rastermap('map.png')

# A = Point(40.0, 40.0)
# B = Point(250.0, 240.0)

# layer.g_lst = [layer.g_lst[0],]

# u = Compute(layer).solve(A, B)

p_gon = layer.g_lst[0].shape

A = Vertex(p_gon, 589)
B = Vertex(p_gon, 118)

u = Compute(layer)
piece, i_lst = u.first_collision(A, B)
r, l = piece.go_through(A, B, i_lst)

plt.figure()
piece.plot()
r.plot()
l.plot()

plt.axis("equal")
plt.grid()
plt.show()