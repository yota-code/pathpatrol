#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

class Piece() :
	def __init__(self, x_lst=None, y_lst=None, m_lst=None) :
		""" the line is expected to turn in the trigonometric way in a direct (Oxy) frame """
		self.x_arr = np.array(x_lst if x_lst is not None else list(), dtype=np.float64)
		self.y_arr = np.array(y_lst if y_lst is not None else list(), dtype=np.float64)

		assert(len(self.x_arr) == len(self.y_arr))

		self.m_arr = np.array(m_lst if m_lst is not None else ( [0,] * len(self) ), dtype=np.int8)
		"""
		1: pour un début de cavité
		2: pour un point du polygone convexe
		-2: pour un point d'une cavité
		-1: pour une fin de cavité
		"""

	def __getitem__(self, i) :
		return self.x_arr[i % len(self)], self.y_arr[i % len(self)]
	
	def __len__(self) :
		return len(self.x_arr)

	def plot(self) :
		plt.plot(self.x_arr, self.y_arr, '+--', color="tab:blue")
		plt.plot(self.x_arr[0], self.y_arr[0], '+', color="tab:red")
		plt.grid()
		plt.axis("equal")
		plt.show()


	def detour(self) :
		"""
		try to cross a cavity, iter each next point after at, which is not already in a cavity
		"""
		z_prev = None
		while (z_curr := np.sum(self.m_arr)) != z_prev :
			a = 0
			while a < (len(self) + 1) :
				b = a + 2
				while self.m_arr[b] < 0 and b < (len(self) + 1) :
					b += 1
					if self.side(self[a], self[b], self[b-1]) < 0 :
						self.m_arr[b-1] = -2
		# 	for a in range(0, len(self) + 1) :
		# 		print(f"     i_{i} = {self.x_arr[i]}, {self.y_arr[i]}")
		# 		j = i + 2

		# 		if self.m_arr[i] == 0 :
		# 			if a is None : # on prend le premier point non encore assigné comme point de départ
		# 				a = i
		# 				print(f"---> a_{a} = {self.x_arr[a]}, {self.y_arr[a]}")
		# 				continue
		# 			elif 1 < i - a  : # si on a avancé de 2 points
		# 				b = i
		# 				print(f"---> b_{b} = {self.x_arr[b]}, {self.y_arr[b]}")
		# 				w = self.side(self[a], self[b], self[b-1])
		# 				print("w = ", w)
		# 				if w > 0 :
		# 					self.m_arr[a+1:b] = -2

		# print(self.m_arr)

	def side(self, A, B, M) :
		""" returns
		0 if M is on AB,
		+1 if M is on the right side of AB
		-1 if M is on the left side of AB
		"""
		ax, ay = A
		bx, by = B
		mx, my = M
		return ((bx - ax)*(my - ay)) - ((by - ay)*(mx - ax))



if __name__ == '__main__' :
	u = Piece(
		[71,92,93,96,97,98,99,101,102,103,104,105,106,109,110,111,112,113,116,116,118,119,119,121,121,122,122,123,124,124,125,125,126,126,127,128,130,131,131,133,134,134,136,136,137,137,138,138,139,139,141,141,142,142,143,143,144,144,145,145,146,146,147,147,148,148,138,138,137,137,136,136,135,135,134,134,133,133,132,132,130,130,129,129,128,128,127,127,125,125,123,122,122,120,119,118,118,117,114,113,112,111,110,109,104,103,100,99,89,88,79,78,75,74,69,68,65,64,63,62,59,57,56,55,55,53,53,52,52,51,51,50,50,49,49,48,48,47,47,48,48,49,49,50,50,51,51,53,53,54,54,55,55,57,58,60,62,62,49,48,46,45,44,43,42,42,41,41,39,39,38,38,37,37,36,36,35,35,34,34,33,33,34,34,35,35,36,36,37,37,38,38,39,39,40,40,41,41,43,43,44,44,45,45,49,49,50,51,52,52,53,54,55,58,59,60,61,64,65,70],
		[23,23,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31,34,35,37,37,38,40,41,42,43,44,44,46,47,48,49,57,58,58,60,60,61,63,63,64,66,67,68,69,70,71,72,73,75,76,77,78,79,80,81,82,83,86,87,90,91,96,97,126,126,106,105,100,99,96,95,92,91,90,89,88,87,86,84,83,82,81,80,79,78,77,75,74,72,72,71,69,69,68,67,66,66,65,65,64,64,63,63,62,62,61,61,60,60,61,61,62,62,63,63,64,64,65,65,67,67,68,69,71,72,73,74,75,76,77,78,79,84,85,86,87,100,101,104,105,106,107,110,111,112,114,115,116,117,118,119,121,121,123,123,134,134,133,133,132,132,131,131,130,129,128,126,125,124,123,122,119,118,115,114,100,99,85,84,72,71,64,63,60,59,58,57,54,53,52,51,50,49,48,47,46,44,41,40,39,38,37,33,32,31,31,30,29,28,28,27,27,26,26,25,25,24,24]
	)
	u.detour()
	u.plot()