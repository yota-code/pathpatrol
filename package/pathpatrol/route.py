#!/usr/bin/env python3

import collections
import enum
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from cc_pathlib import Path

from pathpatrol.sequence import Point, Vertex
from pathpatrol.common import *

"""
un point peut-être libre ou bien attaché à un des bords, dans le sens trigo ou anti trigo
"""

Path("route.log").write_text('')
def log(txt) :
	with Path("route.log").open('at') as fid :
		fid.write(str(txt) + '\n')

class SegmentType(enum.IntEnum) :
	UNKNOWN = 0 # the amount of collisions is unknown and must be computed
	EDGE = 1 # the segment is exactly the edge of a polygon
	FREE = 2 # the segment is not an edge but has been checked and is clear from collisions

class Route() :
	""" a dict of segments
	
	key are a pair of points or vertices
	value is a status
	"""

	def __init__(self) :
		self.r_map = dict()
		self.d_set = set() # a set of already resolved segments

	def push(self, r, status=SegmentType.UNKNOWN) :
		if r in self.d_set :
			log(f">>> RouteMap.push({r}, {status}) :: " + ("SKIPPED"))
			return
		if r not in self.r_map or self.r_map[r] < status :
			log(f">>> RouteMap.push({r}, {status}) :: " + ("UPDATE" if r in self.r_map else "INSERT"))
			self.r_map[r] = status

	def pop(self, r) :
		self.d_set.add(r)
		self.r_map.pop(r, None)

	def __setitem__(self, r, status) :
		self.r_map[r] = status

	def __getitem__(self, r) :
		return self.r_map[r]

	def __iter__(self) :
		for r in sorted(self.r_map, key=lambda r : (r[0].xy, r[1].xy)) :
			yield r, self.r_map[r]

	def __str__(self) :
		s_lst = list()
		for r in sorted(self.r_map, key=lambda x : (x[0].xy, x[1].xy)) :
			s_lst.append(f"{r[0]}\t{r[1]}\t{self.r_map[r]}")
		return '\n'.join(s_lst)
	
	def add_sequence(self, p_lst) :
		""" from a sequence of point, add each pairs as segments """
		A = None
		for B in p_lst :
			if A is not None :
				status = SegmentType.UNKNOWN
				if isinstance(A, Vertex) and isinstance(B, Vertex) and B.p_gon is A.p_gon and abs(A.n - B.n) == 1 :
					status = SegmentType.EDGE
				self.push((A, B), status)
			A = B

	def plot(self, k=16) :
		cmap = matplotlib.colormaps['viridis']
		for (A, B), status in self :
			line = 'solid'
			if isinstance(A, Vertex) and isinstance(B, Vertex) and B.p_gon is A.p_gon and A.n - B.n == 1 :
				line = 'dashed'
			(ax, ay), (bx, by) = A.xy, B.xy
			plt.plot([ax, bx], [ay, by], color={
				SegmentType.UNKNOWN : "tab:red",
				SegmentType.EDGE: "tab:green",
				SegmentType.FREE: "tab:purple"}[status],
				linestyle=line, linewidth=6
			)
			for i in range(k) :
				t0 = i / k
				t1 = (i+1) / k
				plt.plot(
					[ax*(1-t0) + bx*t0, ax*(1-t1) + bx*t1],
					[ay*(1-t0) + by*t0, ay*(1-t1) + by*t1],
					color=cmap(i/(k-1))
				)

