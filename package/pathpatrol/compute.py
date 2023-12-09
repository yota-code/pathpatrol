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
from pathpatrol.route import Route, SegmentType
from pathpatrol.sequence import Point, Vertex
from pathpatrol.common import *

"""
un point peut-être libre ou bien attaché à un des bords, dans le sens trigo ou anti trigo
"""


Path("compute.log").write_text('')

def log(txt) :
	with Path("compute.log").open('at') as fid :
		fid.write(str(txt) + '\n')

def argmax(x):
	return max(range(len(x)), key=lambda i: x[i])
# class Point() :
# 	# free point, or status unknown
# 	def __init__(self, x, y) :
# 		self.x = x
# 		self.y = y

# 	@property
# 	def val(self) :
# 		return (self.x, self.y)

# 	def __repr__(self) :
# 		return f"P{self.val}"

# 	def __hash__(self) :
# 		return hash(self.val)
	
# 	def __eq__(self, other) :
# 		return isinstance(other, Point) and (self.val == other.val)

# class Vertex(Point) :
# 	def __init__(self, layer, p, n) :
# 		self.layer = layer
# 		self.p = p
# 		self.n = n

# 	@property
# 	def val(self) :
# 		return tuple(self.layer[self.p].orig.p_arr[self.n,:])
	
# 	def __repr__(self) :
# 		return f"V{self.val} @ {self.p}/{self.n}"

# class Line() :
# 	def __init__(self) :
# 		self.p_lst = list()

# 	def __iter__(self) :
# 		for p in self.p_lst :
# 			yield p

# 	def push(self, p) :
# 		self.p_lst.append(p)

# 	def get_array(self) :
# 		return np.array([p.val for p in self.p_lst])

# 	def get_polygon(self) :
# 		return Polygon(self.get_array())
	
# 	def __repr__(self) :
# 		return '[\n' + '\n'.join(f"\t{p}" for p in self.p_lst) + '\n]'
	
# class RouteMap() :
# 	def __init__(self) :
# 		self.r_map = dict()

# 	def push(self, A, B, status=None) :
# 		r = (A, B)
# 		if r not in self.r_map or status is not None :
# 			log(f"RouteMap.push({A}, {B}, {status}) new route: {r not in self.r_map}")
# 			self.r_map[r] = status

# 	def pop(self, r) :
# 		self.r_map.pop(r, None)

# 	def __setitem__(self, r, status) :
# 		self.r_map[r] = status

# 	def __getitem__(self, r) :
# 		return self.r_map[r]

# 	def __iter__(self) :
# 		for r in sorted(self.r_map, key=lambda x : (x[0].val, x[1].val)) :
# 			yield r, self.r_map[r]

# 	def __str__(self) :
# 		s_lst = list()
# 		for r in sorted(self.r_map, key=lambda x : (x[0].val, x[1].val)) :
# 			s_lst.append(f"{r[0]}\t{r[1]}\t{self.r_map[r]}")
# 		return '\n'.join(s_lst)

class Compute() :
	def __init__(self, layer) :
		self.layer = layer
		self.route = Route()

	def plot(self) :
		""" plot a route """
		self.layer.plot()
		self.route.plot()

	def solve(self, A, B) :
		""" point de départ, point d'arrivée,
		traverse l'espace:
			sans collision: True,
			collision inconnue: None,
		suit une bordure: False"""

		self.route = Route()
		self.route.push((A, B))

		for i in range(3) :
			self.i = i
			for j, (segment, status) in enumerate(self.route) :
				self.j = j
				log(f"-----  {i} {j}  -----")
				if status is SegmentType.UNKNOWN : # the collision status is unknown, let's investigate
					self.run(* segment)

	def run(self, A, B) :
		log(f">>> Compute.run(({A}, {B}))")

		plt.figure(figsize=(12, 12))
		self.plot()
		plt.grid()
		plt.axis("equal")
		plt.savefig(f"img/{self.i:02d}.{self.j:02d}_{A}-{B}.00.png")
		plt.close()

		# le status des collisions est inconnu, il faut fouiller
		piece, i_lst = self.first_collision(A, B)

		if piece is None : # there is no collision at all, mark as checked and loop
			self.route[(A, B)] = True
			return

		plt.figure(figsize=(12, 12))
		(ax, ay), (bx, by) = A.xy, B.xy
		plt.plot([ax, bx], [ay, by], color="black")
		piece.plot()
		p_gon = piece.shape
		for t, i, w in i_lst :
			plt.plot([ax*(1 - t) + bx*t,], [ay*(1 - t) + by*t,], 'o', color=("tab:red" if w else "tab:green"))
			plt.plot([p_gon.x_arr[i],], [p_gon.y_arr[i],], '^', color=("tab:red" if w else "tab:green"))
		plt.plot([ax,], [ay,], color="tab:blue")
		plt.plot([bx,], [by,], color="tab:orange")
		plt.grid()
		plt.axis("equal")
		plt.savefig(f"img/{self.i:02d}.{self.j:02d}_{A}-{B}.01.png")
		plt.close()

		#if isinstance(A, Point) and isinstance(B, Point) :
		# print(piece.convex.is_inside(A.xy), piece.convex.is_inside(B.xy))
		if piece.convex.is_inside(A.xy) or piece.convex.is_inside(B.xy) :
			r_lst = piece.go_through(A, B, i_lst)
		else :
			r_lst = piece.go_around(A, B)

		log(r_lst[0].b_lst)
		log(r_lst[1].b_lst)
		
		plt.figure(figsize=(12, 12))
		plt.title(f"sequence : {A} > {B}")
		piece.plot()
		plt.plot([ax, bx], [ay, by], color="orange")
		for r in r_lst :
			r.plot()
		plt.grid()
		plt.axis("equal")
		plt.savefig(f"img/{self.i:02d}.{self.j:02d}_{A}-{B}.02.png")
		# if self.i == 1 and self.j == 1 :
		# 	plt.show()
		plt.close()

		for r in r_lst :
			self.route.add_sequence(r)

		log(self.route)

		self.route.pop((A, B))
		
		# we extract the culprit


					
			# 		break
			# 		with Path("route.tsv").open('at') as fid:
			# 			fid.write(f"\n{r[0]}\t{r[1]}\t{self.route[r]}\t ==> pop\n")
			# 		q = r
			# 		rst = self.route[r]
			# 		self.route.pop(r)
			# 		with Path("route.tsv").open('at') as fid:
			# 			fid.write(f"--- {len(self.route.r_map)}\n" + str(self.route) + '\n~~~\n')

			# 		for w in [True, False] :

			# 			r_lin = self.go_around(A, B, p, i_lst, w)
			# 			r_gon = r_lin.get_polygon()
			# 			c_arr = r_gon.convexity()

			# 			with Path("route.tsv").open('at') as fid:
			# 				fid.write(f"\nprocessing {'left' if w else 'right'}\n")
			# 				fid.write(str([r.val for r, c in zip(r_lin, c_arr) if c < math.pi]) + '\n')

			# 			plt.plot(
			# 				[r.val[0] for r in r_lin],
			# 				[r.val[1] for r in r_lin]
			# 			)
			# 			plt.plot(
			# 				[r.val[0] for r, c in zip(r_lin, c_arr) if c < math.pi],
			# 				[r.val[1] for r, c in zip(r_lin, c_arr) if c < math.pi]
			# 			)

			# 			prev = None
			# 			for next, c in zip(r_lin, c_arr) :
			# 				if prev is None :
			# 					prev = next
			# 				elif c < math.pi :
			# 					m = None
			# 					if isinstance(prev, Vertex) and isinstance(next, Vertex) and abs(prev.n - next.n) == 1 :
			# 						m = False
			# 					if q != (prev, next) :
			# 						with Path("route.tsv").open('at') as fid :
			# 							fid.write(f"\n{prev}\t{next}\t{m}\t <== push\n")
			# 						self.route.push(prev, next, m)
			# 						with Path("route.tsv").open('at') as fid:
			# 							fid.write(f"--- {len(self.route.r_map)}\n" + str(self.route) + '\n~~~\n')
			# 					prev = next
			# 		plt.grid()
			# 		plt.axis("equal")
			# 		plt.savefig(f"{z:04d}.b.png")
			# 		#plt.show()
			# 		plt.close()

			# 		break
			# else :
			# 	break
			
			# break
			# z += 1
			# if z >= 32 :
			# 	break


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

	def goaround_shape(self, A, B, p, i_lst, way) :

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
		""" iter on each polygon and try to find one which collide
		
		A and B can be eitther a point or a Vertice.
		"""
		Axy, Bxy = A.xy, B.xy

		for piece in self.layer :
			p_gon = piece.shape
			if not (p_gon.is_inside_box(Axy) or p_gon.is_inside_box(Bxy) or p_gon.box_as_polygon().intersection(Axy, Bxy)) :
				# the segment doesn't even pass through the polygon box, continue
				continue
			i_lst = p_gon.intersection(Axy, Bxy)
			if i_lst :
				return piece, i_lst # i_lst est retourné dans l'ordre des sommets, pas des intersections
		return None, list()
	

	def get_point_type(self, A) :
		""" ne pas utiliser, useless ! """
		p_set = set()
		for p in p_map :
			if p_map[p] is None :
				p_map[p] = dict()
				for i, piece in enumerate(layer) :
					p_map[p][i] = piece.get_point_type(A)
					p_set.append[i]


