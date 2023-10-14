#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pawpatrol.contour import Contour_Cartesian as Contour

class RasterMap() :

	jump_factor = np.array([
		[1, 2, 4],
		[8, 0, 16],
		[32, 64, 128]
	])

	jump_index = {
		11: (0, -1), 15: (0, -1), 22: (-1, 0), 23: (-1, -1), 31: (0, -1), 43: (1, -1), 47: (1, -1),
		63: (1, -1), 104: (1, 0), 105: (1, 0), 107: (1, 0), 111: (1, 0), 150: (-1, 0), 151: (-1, -1),
		159: (0, -1), 208: (0, 1), 212: (-1, 1), 214: (-1, 0), 215: (-1, -1), 232: (1, 1), 233: (1, 1),
		235: (1, 1), 240: (0, 1), 244: (-1, 1), 246: (-1, 0), 248: (0, 1), 249: (0, 1), 252: (-1, 1)
	}

	def __init__(self, lvl) :
		# lvl must be a black & white image, with zero/black for blocked, one/white for clear pixel
		self.lvl = lvl[:].astype(np.uint16)

	def plot(self) :
		for n in self.g_map :
			g_lst = self.g_map[n]
			for c, r in g_lst :
				self.lvl[r, c] = 10
	
		plt.imshow(self.lvl, origin='lower')
		plt.show()

	def fill(self, r, c, n) :
		""" fill with the color n, all the plane which has the same color as the color found at [r, c]
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

	def segment(self) :
		"""colorize the B/W plane with colors from 2 to n"""
		row, col = self.lvl.shape

		self.g_map = dict()
		
		z = [1, 1]
		for r in range(row) :
			for c in range(col) :
				o = self.lvl[r, c]
				if o in [0, 1] :
					n = 2*z[o] + o
					self.fill(r, c, n)
					if o == 0 :
						self.g_map[n] = self.contour(r, c, n)
					z[o] += 1

	def contour(self, r, c, n) :
		"""follow the pixels around a blob,
		return a list of consecutive pixels gathered clockwise"""
		ext = np.zeros([k + 2 for k in self.lvl.shape], dtype=np.uint16) # extended version of lvl with a 1 px border
		ext[1:-1,1:-1] = ( self.lvl == n )

		g_lst = Contour([c,], [r,])
		z = 0
		while True :
			p = ext[r:r+3,c:c+3]
			i, j = self.jump_index[np.sum(p * self.jump_factor)]
			if i == 0 and j == 0 :
				print("ERR: next step unknown\n", p, np.sum(p * self.jump_factor), i, j)
				plt.imshow(self.lvl)
				plt.show()
				sys.exit(0)
			r += i
			c += j
			if (c, r) == g_lst[0] :
				break
			g_lst.push(c, r)
		return g_lst

