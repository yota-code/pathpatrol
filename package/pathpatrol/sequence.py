#!/usr/bin/env python3

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

class Point() :
	# free point
	__slots__ = 'x', 'y'
	def __init__(self, x, y) :
		self.x = x
		self.y = y

	@property
	def xy(self) :
		return (self.x, self.y)

	def __repr__(self) :
		return f"P{self.xy}"

	def __hash__(self) :
		return hash(self.xy)
	
	def __eq__(self, other) :
		return self.xy == other.xy

class Vertex() :
	# point of a polygon
	__slots__ = 'p_gon', 'n'
	def __init__(self, p_gon, n) :
		self.p_gon = p_gon
		self.n = n

	@property
	def xy(self) :
		return self.p_gon[self.n]
	
	def __hash__(self) :
		return hash(self.xy)
	
	def __eq__(self, other) :
		return self.xy == other.xy
	
	def __repr__(self) :
		return f"V{self.xy}@{self.n}"

class Sequence() :
	""" a list of points or vertices """

	def __init__(self) :
		self.b_lst = list()

	def push_point(self, x, y) :
		self.b_lst.append(Point(x, y))

	def push_vertex(self, p_gon, n) :
		self.b_lst.append(Vertex(p_gon, n))

	def push(self, A) :
		if not isinstance(A, (Point, Vertex)) :
			raise ValueError
		self.b_lst.append(A)
		return self
	
	def reversed(self) :
		self.b_lst = self.b_lst[::-1]
		return self

	def push_vertices(self, p_gon, n_from, n_to, w=None) :
		print(f">>> Sequence.push_vertices({id(p_gon)}, {n_from}, {n_to})")
		if w is None :
			w = 1 if n_from < n_to else -1
		for n in range(n_from, n_to+w, w) :
			self.b_lst.append(Vertex(p_gon, n))

	def __iter__(self) :
		for b in self.b_lst :
			yield b

	def plot(self, k=16) :
		cmap = matplotlib.colormaps['viridis']
		A = None
		for B in self :
			if A is not None :
				(ax, ay), (bx, by) = A.xy, B.xy
				for i in range(k) :
					t0 = i / k
					t1 = (i+1) / k
					plt.plot(
						[ax*(1-t0) + bx*t0, ax*(1-t1) + bx*t1],
						[ay*(1-t0) + by*t0, ay*(1-t1) + by*t1],
						color=cmap(i/(k-1)), linewidth=2
					)
			A = B

	def iterative_convex_reduction(self, w=1) :
		while 3 <= len(self.b_lst) :
			xa, ya = self.b_lst[0].xy # A
			xb, yb = self.b_lst[1].xy # B
			ABx, ABy = xb - xa, yb - ya
			for i, b in enumerate(self.b_lst[2:]) :
				xc, yc = self.b_lst[i+2].xy # C
				BCx, BCy = xc - xb, yc - yb
				u = ABx * BCy - ABy * BCx
				if w * u <= 0.0 :
					self.b_lst[i+1] = None
				ABx, ABy, xb, yb = BCx, BCy, xc, yc
			p_len = len(self.b_lst)
			self.b_lst = [b for b in self.b_lst if b is not None]
			print(self.b_lst)
			n_len = len(self.b_lst)
			if p_len == n_len :
				break
		return self
