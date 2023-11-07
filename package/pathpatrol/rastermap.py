#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.polygon import Polygon

class RasterMap() :

	jump_factor = np.array([
		[1, 2, 4],
		[8, 0, 16],
		[32, 64, 128]
	])

	jump_index = { # clockwize : row, col relative to the matrix
		2: (-1, 0), 9: (0, -1), 11: (0, -1), 15: (0, -1), 22: (-1, 0), 23: (-1, -1), 27: (0, -1), 31: (0, -1), 43: (1, -1), 47: (1, -1),
		63: (1, -1), 64: (1, 0), 104: (1, 0), 105: (1, 0), 107: (1, 0), 111: (1, 0), 150: (-1, 0), 151: (-1, -1),
		159: (0, -1), 191: (1, -1), 208: (0, 1), 212: (-1, 1), 214: (-1, 0), 215: (-1, -1), 232: (1, 1), 233: (1, 1),
		235: (1, 1), 240: (0, 1), 244: (-1, 1), 246: (-1, 0), 247: (-1, -1), 248: (0, 1), 249: (0, 1), 252: (-1, 1)
	}

	def __init__(self, lvl) :
		# lvl must be a black & white image, with:
		#   - zero/black for the obstacle,
		#   - one/white for clear pixel
		self.lvl = lvl[:].astype(np.uint16)

	def plot(self) :
		for n in self.g_map :
			g_lst = self.g_map[n]
			for c, r in g_lst :
				self.lvl[r, c] = 10
	
		plt.imshow(self.lvl, origin='lower')
		plt.show()

	def fill(self, r, c, n) :
		""" fill with the color n, the part of the plane which is connected to the initial pixel (r, c)
		modify self.lvl in place """
		row, col = self.lvl.shape
		o = self.lvl[r, c]

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

	def stroke(self, r, c, n) :
		"""follow the pixels around a blob,
		return a list of consecutive pixels gathered clockwise"""
		ext = np.zeros([k + 2 for k in self.lvl.shape], dtype=np.uint16) # extended version of lvl with a 1 px border
		ext[1:-1,1:-1] = (self.lvl == n)

		x_lst, y_lst = [c,], [r,]
		z = 0
		while True :
			p = ext[r:r+3,c:c+3]
			try :
				i, j = self.jump_index[np.sum(p * self.jump_factor)]
				print(p, np.sum(p * self.jump_factor), i, j, r, c)
			except KeyError :
				print("ERR: next step unknown\n", p, np.sum(p * self.jump_factor), r, c)
				plt.imshow(self.lvl)
				plt.plot(x_lst, y_lst, '--', color="tab:red")
				plt.plot([c,], [r,], '+')
				plt.show()
				sys.exit(0)
			r += i
			c += j
			if (c, r) == (x_lst[0], y_lst[0]) :
				break
			x_lst.append(c)
			y_lst.append(r)
		return Polygon(x_lst, y_lst).simplify()

	def extract(self) :
		"""colorize the B/W plane with colors from 2 to n"""
		row, col = self.lvl.shape
		
		z = [1, 1]
		for r in range(row) :
			for c in range(col) :
				o = self.lvl[r, c]
				if o in [0, 1] :
					n = 2*z[o] + o
					self.fill(r, c, n)
					if o == 0 :
						yield self.stroke(r, c, n)
					z[o] += 1

