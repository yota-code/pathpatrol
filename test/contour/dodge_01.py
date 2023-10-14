#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pawpatrol.contour import Contour_Cartesian

u = Contour_Cartesian()
u.load_json(Path('../demo_01/contour_8.json'))

u.dodge((10, 150), (150, 170))