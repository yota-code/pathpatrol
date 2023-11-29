#!/usr/bin/env python3

import collections
import enum
import math
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from cc_pathlib import Path

from pathpatrol.polygon import Polygon
from pathpatrol.common import *

"""
un point peut-être libre ou bien attaché à un des bords, dans le sens trigo ou anti trigo
"""

Path("route.log").write_text('')

def log(txt) :
	with Path("route.log").open('at') as fid :
		fid.write(str(txt) + '\n')

def argmax(x):
	return max(range(len(x)), key=lambda i: x[i])
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
	
	def __repr__(self) :
		return '[\n' + '\n'.join(f"\t{p}" for p in self.p_lst) + '\n]'
	
class RouteMap() :
	def __init__(self) :
		self.r_map = dict()

	def push(self, A, B, status=None) :
		r = (A, B)
		if r not in self.r_map or status is not None :
			log(f"RouteMap.push({A}, {B}, {status}) new route: {r not in self.r_map}")
			self.r_map[r] = status

	def pop(self, r) :
		self.r_map.pop(r, None)

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
		self.route = RouteMap()

	def plot(self) :
		cmap = matplotlib.colormaps['viridis']
		for (A, B), status in self.route :

			col = {None : "tab:red", True: "tab:green", False: "tab:purple"}
			(ax, ay), (bx, by) = A.val, B.val
			for i in range(10) :
				t0 = i / 10
				t1 = (i+1) / 10
				plt.plot(
					[ax*(1-t0) + bx*t0, ax*(1-t1) + bx*t1],
					[ay*(1-t0) + by*t0, ay*(1-t1) + by*t1],
					color=cmap(i/9)
				)

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


	def goaround_corner(self, A, B) :
		plt.figure(figsize=(12, 12))
		p_gon = self.layer[0].orig
		(ax, ay), (bx, by) = A.val, B.val
		plt.text(ax+0.1, ay, "A")
		# for i in range(len(p_gon)) :
		# 	plt.text(p_gon.x_arr[i], p_gon.y_arr[i] + 0.2, str(i))
		plt.text(bx+0.1, by, "B")

		self.layer[0].plot()

		A = self.sidestep(A, B, True)
		B = self.sidestep(B, A, False)

		# d1, B, A = self.sidestep(B, A)

		self.plot()

		# (ax, ay) = A.val
		# plt.plot([ax, bx], [ay, by])

		plt.grid()
		plt.axis("equal")
		plt.show()

		return

		for P, Q, s in zip([A, B], [B, A], [0, -1]) :
			if isinstance(P, Vertex) :
				i_lst = self.collision(P, Q, P.p)
				p_gon = self.layer[0].orig

				t0, i0, w0 = i_lst[s]
				(px, py), (qx, qy) = P.val, Q.val
				plt.text(px+1, py, "P")
				plt.text(qx+1, qy, "Q")

				plt.plot(
					[ax*(1 - t) + bx*t for t, i, w in i_lst],
					[ay*(1 - t) + by*t for t, i, w in i_lst], 'v', color="tab:orange"
				)
				plt.plot(
					[self.layer[0].orig.x_arr[i] for t, i, w in i_lst],
					[self.layer[0].orig.y_arr[i] for t, i, w in i_lst], '^', color="tab:purple"
				)

				print(i0+1, P.n, w0)
				j_lst = list()
				for j in range(P.n+1, i0+1, 1 if P.n < i0 else -1) :
					R = self.layer[P.p].orig[j]
					rx, ry = R
					u = angle_3pt(P.val, Q.val, R)
					j_lst.append(u)
					plt.text(p_gon.x_arr[j], p_gon.y_arr[j], f"{j}:{u:0.3f}")
					plt.plot([rx,], [ry,], 'o', color="tab:green")

				jj = P.n + np.argmax(np.absolute(np.array(j_lst))) + 1

				print(j_lst, w0, jj)


				self.layer[P.p].plot()
				(ax, ay), (bx, by) = A.val, B.val
				plt.plot([ax, bx], [ay, by])


				plt.grid()
				plt.show()

				A = Vertex(self.layer, A.p, jj)


	def push_edge(self, i0, i1, p) :
		""" extract the succession of points which goes from A to B following the egde (A and B must be vertices of the same polygon) """

		log(f">>> follow_edge({A}, {B}")

		if not ( isinstance(A, Vertex) and isinstance(B, Vertex) and A.p == B.p ) :
			return

		p_gon = self.layer[A.p].orig
		r_lin = Line()
		w = 1 if A.n < B.n else -1
		for i in range(A.n, B.n+1, w) : # collect all points form A to B in a line
			r_lin.push(Vertex(self.layer, A.p, i))

		log(f"<-- {r_lin}")

		return r_lin, w

	def push_convex_edge(self, A, B, p, z) :

		""" push a list of points of the egde of the p-th polygon """
		p_gon = self.layer[p].orig
		n_lst = p_gon.get_convex_edge(A.n, B.n)

		prev = n_lst[0]
		for next in n_lst[1:] :
			self.route.push(Vertex(self.layer, p, prev if z else next), Vertex(self.layer, p, next if z else prev), False if abs(prev - next) == 1 else None)
			prev = next

	def sidestep(self, A, B, z) :
		""" décalle le point A pour éviter qu'il ne passe à travers l'obstacle directement à partir de A dans sa route de A vers B
		semble OK
		A et B n'ont pas besoin d'être sur le même polygon ? pourtant ce sera toujours le cas je pense
		"""

		log(f"sidestep({A}, {B}, {z})")

		if isinstance(A, Vertex) : # si A n'est pas un des sommets du polygone, on sort direct
			i_lst = self.collision(A, B, A.p) # compute a list of collisions with the polygon on which A is a Vertex

			if not i_lst :
				# ça serait bien de signaler si on sait directement qu'il n'y a pas de collision entre A et B afin d'accélérer la suite
				return A
			
			p_gon = self.layer[A.p].orig
			t0, i0, z0 = i_lst[0] # return the closest collision

			w = 1 if A.n < i0 else -1
			log(f"\tA.n={A.n} t0={t0} i0={i0} z0={z0} w={w}")

			# on regarde le premier point qui suit, dans la direction de la collision

			M = p_gon[A.n + w]
			u = angle_3pt(A.val, B.val, M)
			# on vérifie que le cas n'est pas trivial, et qu'il n'y aurait pas besoin de faire de sidestep (si AB ne traverse pas directement le polygone en A)
			if z0 : # crossing from right to left
				raise NotImplementedError(f"A={A} B={B} u={u} t0={t0} i0={i0} z0={z0}")
			else :
				if u < 0.0 :
					pass
				else :
					return A

			j_lst = list() # liste des points après A et avant la première collision
			for j in range(A.n + w, i0 + w, w) :
				M = p_gon[j]
				u = angle_3pt(B.val, A.val, M)

				plt.text(p_gon.x_arr[j], p_gon.y_arr[j] + 0.2, f"{j} / {u:0.4f}")

				j_lst.append(abs(u))

			d = argmax(j_lst) + 1
			M = Vertex(self.layer, A.p, A.n + d)

			self.push_convex_edge(A, M, A.p, z)
			if z :
				self.route.push(M, B, None)
				self.route.pop((A, B))
			else :
				self.route.push(B, M, None)
				self.route.pop((B, A))

			return M

	def collision(self, A, B, p) :
		p_gon = self.layer[p].orig
		if not (
			p_gon.is_inside_box(A.val) or
			p_gon.is_inside_box(B.val) or
			p_gon.do_intersect_box(A.val, B.val)
		) :
			return list()
		return sorted(p_gon.scan_intersection(A.val, B.val))

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
	
	def collision(self, A, B, p) :
		p_gon = self.layer[p].orig
		if not (
			p_gon.is_inside_box(A.val) or
			p_gon.is_inside_box(B.val) or
			p_gon.do_intersect_box(A.val, B.val)
		) :
			return list()
		return sorted(p_gon.scan_intersection(A.val, B.val))



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


