#!/usr/bin/env python3

import collections
import itertools
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

Polybox = collections.namedtuple('Polybox', ['left', 'right', 'below', 'above'])


class Polygon() :

	""" a closed line """

	def __init__(self, * pos, is_convex=False) :

		self.is_convex = is_convex

		if len(pos) == 2 :
			x_lst, y_lst = pos
			if len(x_lst) == len(y_lst) :
				self.p_arr = np.array([(x, y) for x, y in zip(x_lst, y_lst)], dtype=np.float64)

		if len(pos) == 1 :
			self.p_arr = pos[0]
		
		self.box = Polybox(
			np.min(self.x_arr), np.max(self.x_arr),
			np.min(self.y_arr), np.max(self.y_arr)
		)

	@property
	def x_arr(self) :
		return self.p_arr[:,0]

	@property
	def y_arr(self) :
		return self.p_arr[:,1]
	
	@property
	def box_array(self) :
		return Polygon(np.array([
			(self.box.left, self.box.below),
			(self.box.right, self.box.below),
			(self.box.right, self.box.above),
			(self.box.left, self.box.above)
		]), is_convex=True)
	
	def to_json(self) :
		return [list(self.x_arr), list(self.y_arr)]

	def __getitem__(self, i) :
		if isinstance(i, int) :
			return self.p_arr[i % len(self),:]
		return self.p_arr[i]
		# elif isinstance(i, slice) :
		# 	return np.array([self[j] for j in range(i.stop)[i]])
		# raise NotImplementedError(f"{type(i)}")
		
	def __iter__(self) :
		for p in self.p_arr :
			yield p
		
	def __repr__(self) :
		return f"Polygon({list(self.x_arr)}, {list(self.y_arr)})"
			
	def __len__(self) :
		return len(self.p_arr)
	
	def iter_segment(self) :
		for i in range(len(self)) :
			yield self[i], self[i+1]

	def iter_boxcorner(self) :
		for cx, cy in self.box_array :
			yield cx, cy

	def convexity(self) :
		""" return for each point the angle blocked by the other points
		if the angle blocked is stricly lower than pi, the point is a convex vertex
		if the angle blocked is higher than tau, the point is enclaved 
		"""
		c_lst = list()
		for i in range(len(self)) :
			u_arr = self.ventilate_vertex(i)
			u_min = np.min(u_arr)
			u_max = np.max(u_arr)
			c_lst.append(u_max - u_min)
		return np.array(c_lst)

	def ventilate_vertex(self, i) :
		""" return the unwrapped angles computed for each points of the polygon
		the first point returned is the first one after i (trigo way)
		"""
		ax, ay = self[i]

		m_arr = np.arctan2(self.y_arr - ay, self.x_arr - ax)
		w_arr = np.unwrap(np.hstack((m_arr[i+1:], m_arr[:i])))

		return w_arr

	def ventilate_point(self, A) :
		""" return the unwrapped angles computed between A and each points of the polygon
		"""
		ax, ay = A

		m_arr = np.arctan2(self.y_arr - ay, self.x_arr - ax)
		w_arr = np.unwrap(m_arr)

		return w_arr

	def plot(self) :
		plt.plot(self.x_arr, self.y_arr, '+--')
		# for i, (x, y) in enumerate(self) :
		# 	plt.text(x+0.1, y+0.1, str(i), color="tab:blue")
		plt.plot([x for x, y in self.iter_boxcorner()], [y for x, y in self.iter_boxcorner()], '-.')

	def is_inside_box(self, A) :
		""" return True if A is outside the box enclosing the polygon """
		ax, ay = A
		return self.box.left <= ax <= self.box.right and self.box.below <= ay <= self.box.above

	def is_obstacle(self, A, B) :
		""" test if the segment A, B is blocked by the polygon """
		(ax, ay), (bx, by) = A, B
		if max(ax, bx) < self.box.left or self.box.right < min(ax, bx) or max(ay, by) < self.box.below or self.box.above < min(ay, by) :
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
	
	def is_inside_shape(self, A) :
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
		for i in range(1, len(self)+1) :
			x0, y0 = self[i-1]
			x1, y1 = self[i]
			move_cur = (x1 - x0, y1 - y0)
			if move_cur == move_pre :
				p_lst[-1] = (p_lst[-1][0] + move_cur[0], p_lst[-1][1] + move_cur[1])
			else :
				p_lst.append(move_cur)
			move_pre = move_cur

		return self.__class__(
			list(itertools.accumulate([x for x, y in p_lst[:-1]])),
			list(itertools.accumulate([y for x, y in p_lst[:-1]]))
		)

	def ventilate(self, M) :
		raise NotImplementedError
		""" return the unwrapped chain of angles around the polynom """
		mx, my = M

		orig = math.atan2(self.y_arr[0] - my, self.x_arr[0] - mx)
		d_arr = np.sqrt((self.x_arr - mx)**2 + (self.y_arr - my)**2)

		m_lst = [orig,] # on risque que la somme soit moins précise
		for i in range(len(self)) :
			j = (i + 1) % len(self)
			(ax, ay), (bx, by) = self.p_arr[i], self.p_arr[j]
			u = ((ax - mx)*(by - my)) - ((ay - my)*(bx - mx))
			m_lst.append(m_lst[-1] + math.asin(u / (d_arr[i] * d_arr[j])))

		return np.array(m_lst)
	
	def to_polar(self, A, B=None) :
		""" return the polygon expressed as an angle (m_arr in radian) and a distance (d_arr)
		from a given point """

		ax, ay = A
		mx, my = self.x_arr, self.y_arr
		
		d_arr = np.sqrt((mx - ax)**2 + (my - ay)**2)
		m_arr = np.unwrap(np.arctan2(my - ay, mx - ax))

		if B is not None :
			bx, by = B
			m_arr -= math.atan2(by - ay, bx - ax)
		
		return m_arr, d_arr

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



