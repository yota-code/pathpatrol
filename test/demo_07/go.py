#!/usr/bin/env python3

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer

u = Layer()
u.load_rastermap('map.png')

"""
on idenfie les arrêtes coupées par le segment et on les classe dans 2 pots
celles de contourne à gauche et celles de contourne à droite
"""

def mods(w_arr) :
	return ((w_arr + math.pi) % math.tau) - math.pi

A = (40, 40)
B = (250, 240)
ax, ay = A
bx, by = B

c_gon = u[0].orig

w_arr = mods((np.arctan2(c_gon.y_arr - ay, c_gon.x_arr - ax) - math.atan2(by - ay, bx - ax)))
plt.plot(w_arr / np.max(np.absolute(w_arr)))
w_arr = (((bx - ax) * (c_gon.y_arr - ay)) - ((by - ay) * (c_gon.x_arr - ax)))
plt.plot(w_arr / np.max(np.absolute(w_arr)))
plt.grid()
plt.show()

i_map = dict()
w_prev = w_arr[0]
for i, w in enumerate(w_arr) :
	if i != 0 :
		if w_prev * w < 0.0 :
			(cx, cy), (dx, dy) = c_gon[i-1], c_gon[i]
			i_map[(i-1, i)] = None
	w_prev = w

"""
il faut maintenant calculer les intersections de chacun de ces segments, les exprimer entre A et B (0.0 et 1.0)
et les trier par ordre croissant
"""

print(i_map)

plt.figure()
u.plot()
plt.plot([40,250], [40, 240])
for i, j in i_map :
	plt.plot([c_gon.x_arr[i], c_gon.x_arr[j]], [c_gon.y_arr[i], c_gon.y_arr[j]], 'v')
plt.grid()
plt.axis("equal")
plt.show()