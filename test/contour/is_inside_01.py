#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pawpatrol.contour import Contour_Cartesian

u = Contour_Cartesian([0.1, 1.1, 0.9, -0.1], [0, 0, 1, 1])

test_lst = [
	[0.5, 0.5, True],
	[1.0, 0.5, False],
	[0.0, 0.5, False],
	[0.5, 0.0, False],
	[0.5, 1.0, False],
]

for x, y, r in test_lst :
	f = u.is_inside((x, y))
	print(x, y, f, r, "\x1b[32mOK\x1b[0m" if f == r else "\x1b[31mKO\x1b[0m", sep='\t')
