#!/usr/bin/env python3

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon

layer = Layer().load_rastermap('map.png')

A = (40, 40)
B = (250, 240)

for piece in layer :
    print(piece.get_point_type(A))
    print(piece.get_point_type(B))

plt.figure(figsize=(16, 16))
layer.plot()
plt.grid()
plt.axis('equal')
plt.savefig("00.png")
plt.close()