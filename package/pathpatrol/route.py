#!/usr/bin/env python3

import collections
import enum
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

class SegTyp(enum.IntEnum):
    UNKNOWN = 0
    CLEAR = 1
    BORDER = 2

Segment = collections.namedtuple('segment', ['from', 'to', 'segtyp'])

class Route() :
	def __init__(self, A, B, p_lst) :
		self.p_lst = p_lst
		self.r_lst = [Segment(A, B, SegTyp.UNKNOWN)]

		while True : # any([r.segtyp == SegTyp.UNKNOWN for r in self.r_lst]) :
			for i, r in enumerate(self.r_lst) :
				if r.segtyp == SegTyp.UNKNOWN :
					w = self.pathfind(r)
					self.r_lst = self.r_lst[:i] + w + self.r_lst[i:]
					break
			else :
				break

