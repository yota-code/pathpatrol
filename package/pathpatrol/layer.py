#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

import pathpatrol.bridge
import pathpatrol.piece

class Layer() :

	c_map = {
		"RR" : "tab:green",
		"LL" : "tab:red",
		"RL" : "tab:blue",
		"LR" : "tab:orange"
	}

	def __init__(self) :
		pass
	
	def load_rastermap(self, img_pth) :
		from pathpatrol.rastermap import RasterMap

		lvl = plt.imread('map.png')[:,:,0].astype(np.uint16)
		r_map = RasterMap(lvl)

		self.g_lst = [pathpatrol.piece.Piece(r.x_arr, r.y_arr) for r in r_map.segment()]

		return self
	
	def compute_route(self) :
		self.r_lst = list()

		for i, j, p, q, w in self.iter_tangent() :
			a, b = self.g_lst[i].convex, self.g_lst[j].convex
			A, B = a[p], b[q]
			for n, g in enumerate(self.g_lst) :
				if n != i and n != j :
					if g.convex.is_blocked(A, B) :
						break
			else :
				self.r_lst.append((i, j, p, q, w))

	def iter_tangent(self) :
		for i in range(len(self.g_lst) - 1) :
			for j in range(i + 1, len(self.g_lst)) :
				a, b = self.g_lst[i].convex, self.g_lst[j].convex
				ai, aj, bi, bj = pathpatrol.bridge.find_crosslines(a, b)
				yield i, j, ai, bi, "LR"
				yield i, j, aj, bj, "RL"
				ai, aj, bi, bj = pathpatrol.bridge.find_outlines(a, b)
				yield i, j, ai, bi, "LL"
				yield i, j, aj, bj, "RR"

	def plot(self) :
		plt.figure()
		for g in self.g_lst :
			g.convex.plot()
		for i, j, p, q, w in self.r_lst :
			a, b = self.g_lst[i].convex, self.g_lst[j].convex
			plt.plot([a[p][0], b[q][0]], [a[p][1], b[q][1]], color=self.c_map[w])
		plt.grid()
		plt.show()