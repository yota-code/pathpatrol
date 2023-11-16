#!/usr/bin/env python3

import collections
import enum
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

"""
un point peut-être libre ou bien attaché à un des bords, dans le sens trigo ou anti trigo

"""


	
class Point() :
	def __init__(self) :
		pass

	def from_coordinates(self, px, py) :
		self.px, self.py = px, py

Segment = collections.namedtuple('segment', ['from', 'to', 'segtyp'])

class Route() :
	def __init__(self, layer) :
		self.layer = layer
		self.route = list()
		while True : # any([r.segtyp == SegTyp.UNKNOWN for r in self.r_lst]) :
			for i, r in enumerate(self.r_lst) :
				if r.segtyp == SegTyp.UNKNOWN :
					w = self.pathfind(r)
					self.r_lst = self.r_lst[:i] + w + self.r_lst[i:]
					break
			else :
				break

	def compute(self, A, B) :
		self.route = [(A, B)]


	def point_from_coordinates(self, px, py) :
		