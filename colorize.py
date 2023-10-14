#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

lvl = plt.imread('demo_01.png')[:,:,0].astype(np.uint16)

jump_factor = np.array([
	[1, 2, 4],
	[8, 0, 16],
	[32, 64, 128]
])

jump_index = [(0, 0),] * 256

jump_index[240] = (0, 1)
jump_index[248] = (0, 1)
jump_index[104] = (1, 0)
jump_index[235] = (1, 1)
jump_index[249] = (0, 1)
jump_index[232] = (1, 1)
jump_index[107] = (1, 0)
jump_index[11] = (0, -1)
jump_index[63] = (1, -1)
jump_index[111] = (1, 0)
jump_index[31] = (0, -1)
jump_index[15] = (0, -1)
jump_index[43] = (1, -1)
jump_index[105] = (1, 0)
jump_index[233] = (1, 1)
jump_index[252] = (-1, 1)
jump_index[246] = (-1, 0)
jump_index[208] = (0, 1)
jump_index[244] = (-1, 1)
jump_index[212] = (-1, 1)
jump_index[214] = (-1, 0)
jump_index[47] = (1, -1)
jump_index[23] = (-1, -1)
jump_index[159] = (0, -1)
jump_index[22] = (-1, 0)
jump_index[215] = (-1, -1)
jump_index[150] = (-1, 0)
jump_index[151] = (-1, -1)

# print(jump_index)

# 1. on construit le polygone convexe autour de chaque obstacle
# 2. si un obstacle rentre dans le polygone convexe d'un autre, on défini un chemin (soit au milieu, soit au plus court, soit autre chose)

class Cartographer() :
	def __init__(self, lvl) :
		# lvl must be a black & white raster, with zero for occupied, one for clear pixel
		self.lvl = lvl.astype(np.uint16)

		self.colorize()

		Path("circle.pson").write_text(repr(self.g_map))

		self.plot_debug()
		# for n in self.g_map :
		# 	for r, c in self.g_map[n] :
		# 		self.lvl[r, c] = 20

	def is_point_in_convex_hull(self, p, h_lst) :
		x_max = max([x for y, x in h_lst]) + 1

	def plot_debug(self) :
		plt.figure()
		for n in self.g_map :
			plt.plot([c for r, c in self.g_map[n]], [r for r, c in self.g_map[n]])

		plt.imshow(self.lvl)
		plt.colorbar()
		for n in self.g_map :
			h_prev = 0
			h_lst = self.g_map[n]
			while len(h_lst) != h_prev :
				h_prev = len(h_lst)
				h_lst = self.convex(h_lst)
			print(len(self.g_map[n]), "->", len(h_lst))
			r_lst = [r for r, c in h_lst]
			c_lst = [c for r, c in h_lst]
			plt.plot(c_lst, r_lst)
			#for r, c in self.g_map[n] :
			#	self.lvl[r, c] = 20

		plt.show()

	def fill(self, o, r, c, n) :
		row, col = self.lvl.shape
		w_set = {(r, c),}

		while w_set :
			r, c = w_set.pop()
			self.lvl[r, c] = n
			for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)] :
				a, b = (r+i, c+j)
				if 0 <= a < row and 0 <= b < col :
					e = self.lvl[a, b]
					if e == o :
						self.lvl[a, b] = n
						w_set.add((a, b))

	def colorize(self) :
		row, col = self.lvl.shape
		self.g_map = dict()

		z = [1, 1]
		for r in range(row) :
			for c in range(col) :
				o = self.lvl[r, c]
				if o in [0, 1] :
					n = 2*z[o] + o
					self.fill(o, r, c, n)
					if o == 0 :
						self.g_map[n] = self.circle(r, c, n)
					z[o] += 1

	def convex(self, g_lst) :
		g_lst = g_lst + [g_lst[0],]#  + [self.g_map[n][0],]
		h_lst = list()

		p = 0
		while p < len(g_lst) :
			h_lst.append(g_lst[p])
			p = self.scan(g_lst, p)

		return h_lst[:-1]
	
	def scan(self, g_lst, start_at) :
		i, j = start_at, start_at + 2		
		while j < len(g_lst) :
			ax, ay = g_lst[i]
			bx, by = g_lst[j]
			for k in range(i + 1, j) :
				mx, my = g_lst[k]
				z = ((bx - ax)*(my - ay)) - ((by - ay)*(mx - ax))
				if z > 0 :
					return j - 1
			else :
				j += 1
		return j - 1

	def circle(self, r, c, n) :
		""" follow the pixels around a blob,
		return a list of consecutive pixels gathered clockwise"""
		ext = np.zeros([k + 2 for k in self.lvl.shape], dtype=np.uint16) # extended version of lvl with a 1 px border
		ext[1:-1,1:-1] = ( self.lvl == n )

		g_lst = [(r, c),]
		while True :
			p = ext[r:r+3,c:c+3]
			i, j = jump_index[np.sum(p * jump_factor)]
			if i == 0 and j == 0 :
				print("ERR: next step unknown\n", p, np.sum(p * jump_factor), i, j)
				plt.imshow(self.lvl)
				plt.show()
				sys.exit(0)
			r += i
			c += j
			if (r, c) == g_lst[0] :
				break
			g_lst.append((r, c))

		return self.simplify(g_lst)

	def simplify(self, g_lst) :
		"""the pixel list contains many redundant points
		this function return a simplified version """

		f_lst = list()
		for a, b in zip(g_lst, g_lst[1:] + [g_lst[0],]) :
			f_lst.append((b[0] - a[0], b[1] - a[1]))

		u_lst = list()
		p_prev = (0, 0)
		p_next = g_lst[0]
		while f_lst :
			p_curr = f_lst.pop(0)
			if p_prev != p_curr :
				u_lst.append(p_next) 
			p_next = (p_next[0] + p_curr[0], p_next[1] + p_curr[1])
			p_prev = p_curr

		return u_lst

u = Cartographer(lvl)


