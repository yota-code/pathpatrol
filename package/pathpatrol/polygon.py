#!/usr/bin/env python3

import collections
import itertools
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

PolyBox = collections.namedtuple("PolyBox", ["left", "right", "bottom", "top"])

class Polygon() :

	""" a closed line """

	def __init__(self, x_lst=None, y_lst=None) :
		""" the line formed by x_lst and y_lst turn in the trigonometric way in a direct (Oxy) frame """
		self.x_arr = np.array(x_lst if x_lst is not None else list(), dtype=np.float64)
		self.y_arr = np.array(y_lst if y_lst is not None else list(), dtype=np.float64)

		assert(len(self.x_arr) == len(self.y_arr))

		self.box = PolyBox(np.min(self.x_arr), np.max(self.x_arr), np.min(self.y_arr), np.max(self.y_arr))

	def __getitem__(self, i) :
		if isinstance(i, int) :
			return self.x_arr[i % len(self)], self.y_arr[i % len(self)]
		elif isinstance(i, slice) :
			return np.array([self[j] for j in range(i.stop)[i]]).T
		raise NotImplementedError(f"{type(i)}")
		
	def __iter__(self) :
		for i in range(len(self)) :
			yield self[i]
		
	def __repr__(self) :
		return f"Polygon({list(self.x_arr)}, {list(self.y_arr)})"
			
	def __len__(self) :
		return len(self.x_arr)
	
	def iter_segment(self) :
		for i in range(len(self)) :
			yield self[i], self[i+1]

	def iter_boxcorner(self) :
		for cx, cy in [
			(self.box.left, self.box.bottom),
			(self.box.right, self.box.bottom),
			(self.box.right, self.box.top),
			(self.box.left, self.box.top)
		] :
			yield cx, cy

	def plot(self) :
		plt.plot(self.x_arr, self.y_arr, '+--')

	def is_blocked(self, A, B) :
		""" test if the segment A, B is blocked by the polygon """
		(ax, ay), (bx, by) = A, B
		if max(ax, bx) < self.box[0] or self.box[1] < min(ax, bx) or max(ay, by) < self.box[2] or self.box[3] < min(ay, by) :
			return False
		for C, D in self.iter_segment() :
			k1, k2 = self.intersection(A, B, C, D)
			if 0.0 <= k1 <= 1.0 and 0.0 <= k2 <= 1.0 :
				return True
		return False

	def intersection(self, A, B, C, D) :
		""" test if AB intersect CD, return the position of the intesection points or Math.Inf, if parrallel """
		(ax, ay), (bx, by) = A, B
		(cx, cy), (dx, dy) = C, D

		d = (ax*cy - ax*dy - ay*cx + ay*dx - bx*cy + bx*dy + by*cx - by*dx)

		if math.isclose(d, 0.0, abs_tol=1e-12) :
			return math.inf, math.inf

		k1 = (ax*cy - ax*dy - ay*cx + ay*dx + cx*dy - cy*dx) / d
		k2 = (-ax*by + ax*cy + ay*bx - ay*cx - bx*cy + by*cx) / d

		return k1, k2
	
	def is_inside(self, A) :
		""" return True if point A is inside the Polygon """
		x_min, x_max, y_min, y_max = self.box

		if not (x_min <= A[0] <= x_max and y_min <= A[1] <= y_max) :
			return False

		B = (x_max + 4, A[1])

		p = 0
		for C, D in self.iter_segment() :
			# on pourrait optimiser, avec un test d'intersection qui tient compte de la situation particulière de A, B, horizontal et B toujours à droite
			k1, k2 = self.intersection(A, B, C, D)
			if k1 == 0.0 :
				return True
			if 0.0 <= k1 <= 1.0 and 0.0 <= k2 <= 1.0 :
				p += 1
		return (p % 2) != 0
	
	def simplify(self) :
		""" useful only for pixel based contours """
		p_lst = [self[0],]
		move_pre = (0, 0)
		for i in range(1, len(self)) :
			x0, y0 = self[i-1]
			x1, y1 = self[i]
			move_cur = (x1 - x0, y1 - y0)
			if move_cur == move_pre :
				p_lst[-1] = (p_lst[-1][0] + move_cur[0], p_lst[-1][1] + move_cur[1])
			else :
				p_lst.append(move_cur)
			move_pre = move_cur

		return self.__class__(
			list(itertools.accumulate([x for x, y in p_lst])),
			list(itertools.accumulate([y for x, y in p_lst]))
		)

	def tangent(self, M) :
		""" find the two indices of the points which are tangent to the polygon and passes through M
		w == 1 rightmost, w == -1 leftmost """

		# on itère et on compte en relatif à partir du point de départ

		z_lst = [0.0,]
		a, A = None, None
		for b, B in enumerate(self) :
			if A is None :
				a, A, nma = b, B, math.sqrt((M[0] - B[0])**2 + (M[1] - B[1])**2)
				continue
			nmb = math.sqrt((M[0] - B[0])**2 + (M[1] - B[1])**2)
			z = math.asin( (((B[0] - M[0])*(A[1] - M[1])) - ((B[1] - M[1])*(A[0] - M[0]))) / (nmb * nma) )
			z_lst.append(z_lst[-1] + z)
			a, A, nma = b, B, nmb

		z_arr = np.array(z_lst)

		return int(np.argmin(z_arr)), int(np.argmax(z_arr))
		# z_min = 
		# z_max = 

		# print(z_min, z_max, self[z_min])
		# plt.figure()
		# self.plot()
		# plt.plot([M[0], self[z_min][0]], [M[1], self[z_min][1]])
		# plt.plot([M[0], self[z_max][0]], [M[1], self[z_max][1]])
		# plt.show()
		# print(z_lst)



