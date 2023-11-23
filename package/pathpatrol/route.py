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
		self.route = [(A, B, None)]

		while True :
			for i, (A, B, status) in enumerate(self.route) :
				if status is None :
					# le status des collisions est inconnu, il faut fouiller
					p, i_map = self.all_collisions(A, B)
					print(p, i_map)
					i_lst = i_map[p]

					plt.figure()
					(ax, ay), (bx, by) = A, B
					plt.plot([ax, bx], [ay, by])
					self.layer[p].plot()
					self.layer[0].plot()
					plt.plot(
						[ax*(1 - t) + bx*t for i, t, w in i_lst],
						[ay*(1 - t) + by*t for i, t, w in i_lst], 'o', color="tab:orange"
					)
					plt.grid()
					plt.show()

					break
			else :
				break

	def go_around(self, A, B, p_gon, i_lst) :
		""" A et B ne doivent pas être dans p_gon ni appartenir à ses arrêtes ou ses sommets"""

		(ax, ay), (bx, by) = A, B
		left_lst, right_lst = [B,], [A,]

		for k in range(len(i_lst) // 2) :
			i0, t0, w0 = i_lst[2*k]
			i1, t1, w1 = i_lst[2*k+1]

			if w0 is False and w1 is True :
				m = left_lst
			elif w0 is True and w1 is False :
				m = right_lst
			else :
				raise ValueError
			

			!!! le problème là c'est qu'il ne faut pas perdre l'information de savoir si c'est un pont (qu'il faudra à nouveau tester pour les collisions) ou s'il s'agit d'une arrête
			surtout pendant l'étape suivant de simplification/convexification
			
			m.append((ax*(1 - t0) + bx*t0, ay*(1 - t0) + by*t0))
			for j in range(i0+1, i1+1) :
				m.append((p_gon.x_arr[j], p_gon.y_arr[j]))
			m.append((ax*(1 - t1) + bx*t1, ay*(1 - t1) + by*t1))

		left_lst.append(A)
		right_lst.append(B)




	def all_collisions(self, A, B) :
		i_map = dict()
		first = (math.inf, None)
		for p, piece in enumerate(self.layer) :
			if not (piece.orig.is_inside_box(A) or piece.orig.is_inside_box(B) or piece.orig.do_intersect_box(A, B)) :
				continue
			i_lst = piece.orig.scan_intersection(A, B)
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



		