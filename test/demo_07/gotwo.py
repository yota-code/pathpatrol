#!/usr/bin/env python3

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon

u = Layer()
u.load_rastermap('map.png')

plt.figure()
u.plot()
plt.grid()
plt.axis("equal")
plt.show()

