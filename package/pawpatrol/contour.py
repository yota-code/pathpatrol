#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

class Contour() :
	def __init__(self, x_lst=None, y_lst=None, is_convex=False) :
		""" the line is expected to turn in the trigonometric way in a direct (Oxy) frame """
		self.x_lst = x_lst if x_lst is not None else list()
		self.y_lst = y_lst if y_lst is not None else list()

		assert(len(self.x_lst) == len(self.y_lst))

		self.is_convex = is_convex

	def save_json(self, pth) :
		pth.save({
			'x' : self.x_lst,
			'y' : self.y_lst,
			'is_convex' : self.is_convex
		})

	def load_json(self, pth) :
		k = pth.load()
		self.x_lst = k['x']
		self.y_lst = k['y']
		self.is_convex = k['is_convex']

	def __repr__(self) :
		return f"{self.__class__.__name__}({self.x_lst}, {self.y_lst}) {len(self.x_lst)}"

	def push(self, x, y) :
		self.x_lst.append(x)
		self.y_lst.append(y)

	def __getitem__(self, i) :
		if isinstance(i, int) :
			return self.x_lst[i % len(self)], self.y_lst[i % len(self)]
		else :
			raise NotImplementedError

	def get_arc(self, start:int, stop:int, step:int = 1) :
		p_lst = list()
		i = start
		while i != stop :
			p_lst.append((self.x_lst[i], self.y_lst[i]))
			i = (i + step) % len(self)
		p_lst.append((self.x_lst[i], self.y_lst[i]))
		return p_lst

	def __iter__(self) :
		for x, y in zip(self.x_lst, self.y_lst) :
			yield x, y

	def segments(self) :
		for i in range(len(self)) :
			yield self[i], self[i+1]

	def __len__(self) :
		return len(self.x_lst)

	def convex(self) :
		size = 0
		line = self
		while len(line) != size :
			size = len(line)
			line = line._convex_pass()

		line.is_convex = True

		return line

	def _convex_pass(self) :
		""" does one iteration to find the convex hull, return a new Contour """
		u = self.__class__()

		p = 0
		while p < (len(self) + 1) :
			u.push(* self[p])
			p = self._convex_scan(p)

		return u

	def _convex_scan(self, start_at) :
		i, j = start_at, start_at + 2		
		while j < (len(self) + 1) :
			for k in range(i + 1, j) :
				z = self.side(self[i], self[j], self[k])
				if z < 0 :
					# one point found outside the convex hull
					return j - 1
			else :
				j += 1
		return j - 1

class Contour_Cartesian(Contour) :
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

	def dodge(self, A, B) :
		p_min, p_max = 0.0, 0.0
		i_min, i_max = None, None

		n_ab = np.sqrt((B[0] - A[0])**2 + (B[1] - A[1])**2)

		for i, M in enumerate(self) :
			n_am = np.sqrt((M[0] - A[0])**2 + (M[1] - A[1])**2)
			z = self.side(A, B, M) / (n_am * n_ab)
			if 0 <= p_max < z :
				p_max = z
				i_max = i
			if z < p_min <= 0 :
				p_min = z
				i_min = i

		return i_min, i_max

		print(i_min, i_max)

	def around(self, A, B) :
		p, q = self.dodge(A, B)
		r, s = self.dodge(B, A)

		if p is None or q is None or r is None or s is None :
			print("direct")
			return [A, B]
		else :
			return [
				[A,] + self.get_arc(q, r, -1) + [B,],
				[A,] + self.get_arc(p, s) + [B,]
			]

	def is_inside(self, A) :
		""" does the segment AB goes through the contour """
		x_min, x_max = min(self.x_lst), max(self.x_lst)
		y_min, y_max = min(self.y_lst), max(self.y_lst)

		if not (x_min < A[0] < x_max and y_min < A[1] < y_max) :
			return False

		B = (x_max + 1, A[1])

		p = 0
		for C, D in self.segments() :
			k1, k2 = self.intersection(A, B, C, D)
			if 0 < round(k1, 6) < 1 and 0 < round(k2, 6) < 1 :
				p += 1
			print(A, B, C, D, "k", round(k1, 6), round(k2, 6), 0 < round(k1, 6) < 1 and 0 < round(k2, 6) < 1, p)
		return (p % 2) != 0

	def position(self, A, B, M) :
		""" return the position of M on the segment AB """
		(ax, ay), (bx, by), (mx, my) = A, B, M

		h = (bx - ax)*(mx - ax) + (by - ay)*(my - ay)
		n_ab = math.sqrt((bx - ax)**2 + (by - ay)**2)
		n_am = math.sqrt((mx - ax)**2 + (my - ay)**2)

		if math.isclose(abs(h), n_ab * n_am) :
			return h / n_ab
		
		return math.inf

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





