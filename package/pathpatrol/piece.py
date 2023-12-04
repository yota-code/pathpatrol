#!/usr/bin/env python3

import enum
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.polygon import Polygon
from pathpatrol.sequence import Sequence
# from pathpatrol.common import PointTyp

class PointPos(enum.Enum) :
	UNKNOWN = 0
	OUTSIDE = 1
	CRATER = 2
	CAVITY = 3
	EDGE = 4
	VERTEX = 5
	INSIDE = 6

class Piece() :
	def __init__(self, shape:Polygon) :
		self.shape = shape
		self._pre_compute()

	def _pre_compute(self) :
		"""
		compute the convex hull store it under self.convex (as an instance of Polygon)
		compute also the concavities and if they contains blind points
		"""
		n_arr = np.zeros((len(self),), dtype=np.double)

		for i in range(len(self)) :
			u_arr = self.shape.ventilate_vertex(i)
			n_arr[i] = np.nanmax(u_arr) - np.nanmin(u_arr)

		c_arr = np.where(n_arr < math.pi, 1, 0)

		self.convex = Polygon(self.shape.p_arr[c_arr != 0,:], is_convex=True)

		self.concave = list()
		for i in range(0, len(self)) :
			m = c_arr[(i+1) % len(self)] - c_arr[i]
			if math.tau <= n_arr[i] :
				c = True
			if m < 0 :
				a = i
				c = False
			if m > 0 :
				b = i + 1
				self.concave.append((a, b, c))

	def __len__(self) :
		return len(self.shape)
	
	def to_json(self) :
		return {
			'shape' : self.shape.to_json(),
			'convex' : self.convex.to_json(),
			'concave' : list(self.concave),
		}
	
	def plot(self) :
		plt.fill(self.shape.x_arr, self.shape.y_arr, color="tab:blue", alpha=0.5)
		plt.plot(self.shape.x_arr, self.shape.y_arr, "+--", color="tab:blue", alpha=0.5)
		plt.plot(self.convex.x_arr, self.convex.y_arr, '+--', color="tab:green", linewidth=2, alpha=0.5)
		plt.plot(self.shape.x_arr[0], self.shape.y_arr[0], '+', color="tab:red")
		for a, b, c in self.concave :
			plt.plot(self.shape.x_arr[a:b+1], self.shape.y_arr[a:b+1], color=("tab:red" if c else "tab:orange"))
	
	def get_point_type(self, A) :
		""" return the type of point relative to this piece """

		if not self.shape.is_inside_box(A) :
			# the point is oustside the box 
			return PointPos.OUTSIDE, None
		
		w_arr = self.shape.ventilate_point(A)
		angle = np.max(w_arr) - np.min(w_arr)

		if angle < math.pi :
			# the point is outside the convex hull
			return PointPos.OUTSIDE, None
		else :
			# the point is either in a cavity or inside the shape
			for i, (a, b, c) in enumerate(self.concave) :
				e_gon = Polygon(self.shape[a:b+1])
				if e_gon.is_inside_box(A) and e_gon.is_inside_shape(A) :
					return PointPos.CRATER if angle < math.tau else PointPos.CAVITY, i

			return PointPos.INSIDE, None
		
	def go_around(self, A, B) :
		"""
		A and B must be checked to be points (not vertices) and outiside the convex hull of the piece
		"""

		# no need for intersections, we go for the full ventilation over the convex polygon
		# works only on the convex polygon

		a_vnt = self.convex.ventilate_point(A.xy)
		a_lft = a_vnt.argmax()
		a_rgt = a_vnt.argmin()

		b_vnt = self.convex.ventilate_point(B.xy)
		b_lft = b_vnt.argmin()
		b_rgt = b_vnt.argmax()

		l_seq = Sequence()
		l_seq.push_one(A)
		l_seq.push_vertices(self.convex, a_lft, b_lft, -1)
		l_seq.push_one(B)

		r_seq = Sequence()
		r_seq.push_one(A)
		r_seq.push_vertices(self.convex, a_rgt, b_rgt, 1)
		r_seq.push_one(B)

		return l_seq, r_seq

	def go_through(self, A, B, i_lst) :
		""" cleanest version yet"""

		c'est là qu'il faut dégager les intersections incarnées... mais comment ...

		r_seq, l_seq = Sequence().push(A), Sequence().push(B)
		r_seq, l_seq = Sequence(), Sequence()

		if len(i_lst) % 2 == 0 :
			# standard case, with as many enter as exit
			for (t0, i0, w0), (t1, i1, w1) in zip(i_lst[::2], i_lst[1::2]) :
				print(t0, i0, w0, t1, i1, w1)
				if w0 is False and w1 is True :
					m_seq = l_seq
				elif w0 is True and w1 is False :
					m_seq = r_seq
				else :
					raise ValueError
				m_seq.push_vertices(self.shape, i0+1, i1)

		if r_seq.b_lst :
			r_seq.b_lst = [A,] + r_seq.b_lst  + [B,]
			r_seq.iterative_convex_reduction()
		if l_seq.b_lst :
			l_seq.b_lst = [B,] + l_seq.b_lst  + [A,]
			l_seq.iterative_convex_reduction()

		return r_seq, l_seq.reversed()

	def carved_hull(self, c_lst) :
		""" return the convex hull with only some special shapes carved, according to the list of cavities/craters c_lst"""
		pass

	def _prep_sort(self) :
		self.convex = Polygon(
			self.shape.x_arr[np.where(self.m_arr== 0)],
			self.shape.y_arr[np.where(self.m_arr== 0)]
		)
		self.cavity = dict()
		for i in range(0, len(self)) :
			m = self.m_arr[(i+1) % len(self)] - self.m_arr[i % len(self)]
			if m > 0 :
				a = i
			if m < 0 :
				b = i + 2
				self.cavity[(a, b)] = Polygon(self.shape[a:b,:])

	def _prep_cut(self) :
		"""
		try to cross a cavity, iter each next point after at, which is not already in a cavity
		"""
		def next_zero(start) :
			for i in range(start+1, len(self) + 1) :
				if self.m_arr[i % len(self)] == 0 :
					return i
			raise ValueError

		z_prev = None
		while (z_curr := np.sum(self.m_arr)) != z_prev :
			i = 0
			while True :
				try :
					a = next_zero(i)
					m = next_zero(a)
					b = next_zero(m)
				except ValueError :
					print("end of line")
					break
				w = self.side(self.shape[a], self.shape[b], self.shape[m])
				if w >= 0 :
					self.m_arr[a+1:b] = -1
				i = a
			z_prev = z_curr		

	def side(self, A, B, M) :
		""" returns
		0 if M is on AB,
		+1 if M is on the right side of AB
		-1 if M is on the left side of AB
		"""
		ax, ay = A
		bx, by = B
		mx, my = M
		return ((bx - ax)*(my - ay)) - ((by - ay)*(mx - ax))

if __name__ == '__main__' :
	u = Piece(Polygon(
		[71,92,93,96,97,98,99,101,102,103,104,105,106,109,110,111,112,113,116,116,118,119,119,121,121,122,122,123,124,124,125,125,126,126,127,128,130,131,131,133,134,134,136,136,137,137,138,138,139,139,141,141,142,142,143,143,144,144,145,145,146,146,147,147,148,148,138,138,137,137,136,136,135,135,134,134,133,133,132,132,130,130,129,129,128,128,127,127,125,125,123,122,122,120,119,118,118,117,114,113,112,111,110,109,104,103,100,99,89,88,79,78,75,74,69,68,65,64,63,62,59,57,56,55,55,53,53,52,52,51,51,50,50,49,49,48,48,47,47,48,48,49,49,50,50,51,51,53,53,54,54,55,55,57,58,60,62,62,49,48,46,45,44,43,42,42,41,41,39,39,38,38,37,37,36,36,35,35,34,34,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,43,43,44,44,45,45,49,49,50,51,52,52,53,54,55,58,59,60,61,64,65,70],
		[23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,34,35,37,37,38,40,41,42,43,44,44,46,47,48,49,57,58,58,60,60,61,63,63,64,66,67,68,69,70,71,72,73,75,76,77,78,79,80,81,82,83,86,87,90,91,96,97,126,126,106,105,100,99,96,95,92,91,90,89,88,87,86,84,83,82,81,80,79,78,77,75,74,72,72,71,69,69,68,67,66,66,65,65,64,64,63,63,62,62,61,61,60,60,61,61,62,62,63,63,64,64,65,65,67,67,68,69,71,72,73,74,75,76,77,78,79,84,85,86,87,100,101,104,105,106,107,110,111,112,114,115,116,117,118,119,121,121,123,123,134,134,133,133,132,132,131,131,130,129,128,126,125,124,123,122,119,118,115,114,100,99,85,84,72,71,64,63,60,59,58,57,54,53,52,51,50,49,48,47,46,44,41,40,39,38,37,33,32,31,31,30,29,28,28,27,27,26,26,25,25,24,24]
	))
	#u.detour()
	u.plot()