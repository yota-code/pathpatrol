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

	def __repr__(self) :
		return f"P{self.val}"

	def __hash__(self) :
		return hash(self.val)
	
	def __eq__(self, other) :
		return isinstance(other, Point) and (self.val == other.val)

class Vertex(Point) :
	def __init__(self, layer, p, n) :
		self.layer = layer
		self.p = p
		self.n = n

	@property
	def val(self) :
		return tuple(self.layer[self.p].orig.p_arr[self.n,:])
	
	def __repr__(self) :
		return f"V{self.val} @ {self.p}/{self.n}"

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
	
class RouteMap() :
	def __init__(self) :
		self.r_map = dict()

	def push(self, A, B, status=None) :
		r = (A, B)
		if r not in self.r_map or status is not None :
			self.r_map[r] = status

	def pop(self, r) :
		del self.r_map[r]

	def __setitem__(self, r, status) :
		self.r_map[r] = status

	def __getitem__(self, r) :
		return self.r_map[r]

	def __iter__(self) :
		for r in sorted(self.r_map, key=lambda x : (x[0].val, x[1].val)) :
			yield r, self.r_map[r]

	def __str__(self) :
		s_lst = list()
		for r in sorted(self.r_map, key=lambda x : (x[0].val, x[1].val)) :
			s_lst.append(f"{r[0]}\t{r[1]}\t{self.r_map[r]}")
		return '\n'.join(s_lst)

class Route() :
	def __init__(self, layer) :
		self.layer = layer
		self.route = list()

	def compute(self, from_point, to_point) :
		""" point de départ, point d'arrivée,
		traverse l'espace:
			sans collision: True,
			collision inconnue: None,
		suit une bordure: False"""

		self.route = RouteMap()
		self.route.push(Point(* from_point), Point(* to_point))

		z = 0
		Path("route.tsv").write_text('')

		while True :

			with Path("route.tsv").open('at') as fid:
				fid.write(f"\n======== {z} ========\n")
				fid.write(str(self.route) + '\n')

			plt.figure(figsize=(12, 12))
			self.layer[0].plot()
			for (P, Q), rst in self.route :
				col = {None : "tab:red", True: "tab:green", False: "tab:purple"}
				(ax, ay), (bx, by) = P.val, Q.val
				plt.plot([ax, bx], [ay, by], color=col[rst])
			plt.grid()
			plt.axis("equal")
			plt.savefig(f"{z:04d}.a.png")
			plt.close()
			#plt.show()

			for r, status in self.route :
				if status is None :
					A, B = r
					with Path("route.tsv").open('at') as fid:
						fid.write(f"\n{r[0]}\t{r[1]}\t{self.route[r]}\t !! processing\n")

					# le status des collisions est inconnu, il faut fouiller
					p, i_map = self.all_collisions(A, B)

					if p is None :
						# there is not collision at all, mark as checked and loop
						with Path("route.tsv").open('at') as fid:
							fid.write(f"no collision\n")
						self.route[r] = True
						break

					p_gon = self.layer[p].orig

					# self.goaround_in(self, A, B, p) # ! to be implemented


					"""
					avant de passer au contournement, il faut vérifier que le vecteur de collition ne part pas d'un sommet, directement à travers l'obstacle
					dans ce cas, il faut faire un traitement particulier, et faire glisser les points le long du polygone jusqu'à obtenir une trajectoire qui ne couple plus directment
					elle peut encore couper ailleurs... comment travailler sa convexité ? je ne sais pas
					"""


					i_lst = i_map[p]

					plt.figure(figsize=(12, 12))
					(ax, ay), (bx, by) = A.val, B.val
					plt.plot([ax, bx], [ay, by])
					self.layer[p].plot()
					plt.plot(
						[ax*(1 - t) + bx*t for i, t, w in i_lst],
						[ay*(1 - t) + by*t for i, t, w in i_lst], 'o', color="tab:orange"
					)

					with Path("route.tsv").open('at') as fid:
						fid.write(f"\n{r[0]}\t{r[1]}\t{self.route[r]}\t ==> pop\n")
					q = r
					rst = self.route[r]
					self.route.pop(r)
					with Path("route.tsv").open('at') as fid:
						fid.write(f"--- {len(self.route.r_map)}\n" + str(self.route) + '\n~~~\n')

					for w in [True, False] :

						r_lin = self.go_around(A, B, p, i_lst, w)
						r_gon = r_lin.get_polygon()
						c_arr = r_gon.convexity()

						with Path("route.tsv").open('at') as fid:
							fid.write(f"\nprocessing {'left' if w else 'right'}\n")
							fid.write(str([r.val for r, c in zip(r_lin, c_arr) if c < math.pi]) + '\n')

						plt.plot(
							[r.val[0] for r in r_lin],
							[r.val[1] for r in r_lin]
						)
						plt.plot(
							[r.val[0] for r, c in zip(r_lin, c_arr) if c < math.pi],
							[r.val[1] for r, c in zip(r_lin, c_arr) if c < math.pi]
						)

						prev = None
						for next, c in zip(r_lin, c_arr) :
							if prev is None :
								prev = next
							elif c < math.pi :
								m = None
								if isinstance(prev, Vertex) and isinstance(next, Vertex) and abs(prev.n - next.n) == 1 :
									m = False
								if q != (prev, next) :
									with Path("route.tsv").open('at') as fid :
										fid.write(f"\n{prev}\t{next}\t{m}\t <== push\n")
									self.route.push(prev, next, m)
									with Path("route.tsv").open('at') as fid:
										fid.write(f"--- {len(self.route.r_map)}\n" + str(self.route) + '\n~~~\n')
								prev = next

						
					plt.grid()
					plt.axis("equal")
					plt.savefig(f"{z:04d}.b.png")
					#plt.show()
					plt.close()

					break
			else :
				break
			

			z += 1
			if z >= 32 :
				break


	def goaround_in(self, A, B) :
		plt.figure(figsize=(12, 12))
		self.layer[0].plot()
		(ax, ay), (bx, by) = A.val, B.val
		plt.plot([ax, bx], [ay, by])
		plt.grid()
		plt.show()



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


