#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

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

		for n in self.g_map :
			for r, c in self.g_map[n] :
				self.lvl[r, c] = 20

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

	def circle(self, r, c, n) :

		# r, c = self.g_map[n]

		ext = np.zeros([k + 2 for k in self.lvl.shape], dtype=np.uint16) # extended version of lvl with a 1 px border
		ext[1:-1,1:-1] = ( self.lvl == n )

		r_lst = [(r, c),]
		while True :
			p = ext[r:r+3,c:c+3]
			i, j = jump_index[np.sum(p * jump_factor)]

			if i == 0 and j == 0 :
				print(p, np.sum(p * jump_factor), i, j)
				plt.imshow(self.lvl)
				plt.show()
				sys.exit(0)

			r += i
			c += j
			# self.lvl[r, c] = 0
			if (r, c) == r_lst[0] :
				break
			r_lst.append((r, c))

		#print(r_lst)
		#plt.imshow(self.lvl)
		#plt.show()
		return r_lst

u = Cartographer(lvl)
# u.circle(8)
print(u.g_map)

plt.imshow(u.lvl)
plt.colorbar()
plt.show()

