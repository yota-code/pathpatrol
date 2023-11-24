#!/usr/bin/env python3

import collections
import enum
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.polygon import Polygon

"""
un point peut-être libre ou bien attaché à un des bords, dans le sens trigo ou anti trigo

"""

class Point() :
	# free point, or status unknown
	def __init__(self, x, y) :
		self.x = x
		self.y = y

	@property
	def val(self) :
		return (self.x, self.y)

class Vertex() :
	def __init__(self, layer, p, n) :
		self.layer = layer
		self.p = p
		self.n = n

	@property
	def val(self) :
		return self.layer[self.p].orig.p_arr[self.n,:]

class Line() :
	def __init__(self) :
		self.p_lst = list()

	def __iter__(self) :
		for p in self.p_lst :
			yield p

	def push(self, p) :
		self.p_lst.append(p)

	def get_array(self) :
		return np.array([p.val for p in self.p_lst])

	def get_polygon(self) :
		return Polygon(self.get_array())

class Route() :
	def __init__(self, layer) :
		self.layer = layer
		self.route = list()

	def compute(self, A, B) :
		""" point de départ, point d'arrivée,
		traverse l'espace:
			sans collision: True,
			collision inconnue: None,
		suit une bordure: False"""
		self.route = [(Point(* A), Point(* B), None)]

		while True :
			for i, (A, B, status) in enumerate(self.route) :
				if status is None :
					# le status des collisions est inconnu, il faut fouiller
					p, i_map = self.all_collisions(A, B)
					p_gon = self.layer[p].orig
					i_lst = i_map[p]

					plt.figure()
					(ax, ay), (bx, by) = A.val, B.val
					plt.plot([ax, bx], [ay, by])
					self.layer[p].plot()
					self.layer[0].plot()
					plt.plot(
						[ax*(1 - t) + bx*t for i, t, w in i_lst],
						[ay*(1 - t) + by*t for i, t, w in i_lst], 'o', color="tab:orange"
					)


					for w in [True, False] :
						r_lin = self.go_around(A, B, p, i_lst, w)
						r_gon = r_lin.get_polygon()
						c_arr = r_gon.convexity()

						plt.plot(
							[r.val[0] for r in r_lin],
							[r.val[1] for r in r_lin]
						)
						plt.plot(
							[r.val[0] for r, c in zip(r_lin, c_arr) if c < math.pi],
							[r.val[1] for r, c in zip(r_lin, c_arr) if c < math.pi]
						)

					plt.grid()
					plt.axis("equal")
					plt.show()

					break
			else :
				break
			break

	def go_around(self, A, B, p, i_lst, way) :
		""" A et B ne doivent pas être dans p_gon ni appartenir à ses arrêtes ou ses sommets
		i_lst c'est la liste des points d'intersection
		way est égal à True pour un contournement par la droite, False pour la gauche
		"""
		r_lin = Line()
		r_lin.push(B if way else A)
		for k in range(len(i_lst) // 2) :
			i0, t0, w0 = i_lst[2*k]
			i1, t1, w1 = i_lst[2*k+1]
			if (w0 ^ way) is True and (w1 ^ way) is False :		
				for i in range(i0+1, i1+1) :
					r_lin.push(Vertex(self.layer, p, i))
		r_lin.push(A if way else B)
		if way :
			r_lin.p_lst = r_lin.p_lst[::-1]
		return r_lin

	def all_collisions(self, A, B) :
		i_map = dict()
		first = (math.inf, None)
		for p, piece in enumerate(self.layer) :
			if not (
				piece.orig.is_inside_box(A.val) or
				piece.orig.is_inside_box(B.val) or
				piece.orig.do_intersect_box(A.val, B.val)
			) :
				continue
			i_lst = piece.orig.scan_intersection(A.val, B.val)
			if i_lst :
				i_map[p] = i_lst
				t_min = min(i[1] for i in i_lst)
				if t_min < first[0] :
					first = (t_min, p)
		return first[1], i_map
	
	def first_collision(self, A, B) :
		for p, piece in enumerate(self.layer) :
			if not (piece.orig.is_inside_box(A) or piece.orig.is_inside_box(B) or piece.orig.do_intersect_box(A, B)) :
				continue
			i_lst = piece.orig.scan_intersection(A, B)
			if i_lst :
				return p, i_lst

	def get_point_type(self, A) :
		""" ne pas utiliser, useless ! """
		p_set = set()
		for p in p_map :
			if p_map[p] is None :
				p_map[p] = dict()
				for i, piece in enumerate(layer) :
					p_map[p][i] = piece.get_point_type(A)
					p_set.append[i]



		