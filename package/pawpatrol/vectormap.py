#!/usr/bin/env python3

class VectorMap() :
	def __init__(self, g_map) :
		self.g_map = g_map

		self.h_map = dict()
		for n in g_map :
			self.h_map[n] = g_map[n].convex()
	
