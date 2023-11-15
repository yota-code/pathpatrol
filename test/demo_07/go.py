#!/usr/bin/env python3

import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon

u = Layer()
u.load_rastermap('map.png')

def plot_convexity(p_gon) :
	plt.plot(p_gon.convexity())
	plt.axhline(math.pi)
	plt.axhline(math.tau)


"""
on idenfie les arrêtes coupées par le segment et on les classe dans 2 pots
celles de contourne à gauche et celles de contourne à droite
"""

def mods(w_arr) :
	return ((w_arr + math.pi) % math.tau) - math.pi

A = (40, 40)
B = (250, 240)

(ax, ay), (bx, by) = A, B

z_gon = u[0].orig

w_arr = ((z_gon.x_arr - ax)*(by - ay)) - ((z_gon.y_arr - ay)*(bx - ax))

i_lst = list()
w_prev = w_arr[0]
for i, w in enumerate(w_arr[1:]) :
	if w_prev * w < 0.0 :
		(cx, cy), (dx, dy) = z_gon[i], z_gon[i+1]
		d = (bx - ax)*(dy - cy) - (by - ay)*(dx - cx)
		t = (ax*(cy - dy) + ay*(-cx + dx) + cx*dy - cy*dx) / d
		if 0.0 <= t <= 1.0 :
			i_lst.append((i, t, True if w > 0 else False))
	w_prev = w

""" on doit avoir un nombre pair de traversées franches, si A et B sont bien en dehors du polygone """
assert(len(i_lst) % 2 == 0)

""" on fourre les points dans deux listes, une de contournement à gauche, l'autre à droite"""

left_lst, right_lst = [B,], [A,]

for k in range(len(i_lst) // 2) :
	i0, t0, w0 = i_lst[2*k]
	i1, t1, w1 = i_lst[2*k+1]

	if w0 is False and w1 is True :
		m = left_lst
	elif w0 is True and w1 is False :
		m = right_lst
	else :
		raise ValueError
	
	m.append((ax*(1 - t0) + bx*t0, ay*(1 - t0) + by*t0))
	for j in range(i0+1, i1+1) :
		m.append((z_gon.x_arr[j], z_gon.y_arr[j]))
	m.append((ax*(1 - t1) + bx*t1, ay*(1 - t1) + by*t1))

left_lst.append(A)
right_lst.append(B)

left_arr = np.array(left_lst)
right_arr = np.array(right_lst)

left_gon = Polygon(p_arr=left_arr)
right_gon = Polygon(p_arr=right_arr)

l_gon = Polygon(p_arr=left_arr[left_gon.convexity() < math.pi])
r_gon = Polygon(p_arr=right_arr[right_gon.convexity() < math.pi])

"""
il faut maintenant calculer les intersections de chacun de ces segments, les exprimer entre A et B (0.0 et 1.0)
et les trier par ordre croissant
"""

plt.figure()
u.plot()
plt.plot([40,250], [40, 240])

for i, t, w in i_lst :
	(cx, cy), (dx, dy) = z_gon[i], z_gon[i+1]
	plt.plot([cx, dx], [cy, dy], 'v', color='tab:green' if w else 'tab:red')
plt.plot(
	[ax*(1 - t) + bx*t for i, t, w in i_lst],
	[ay*(1 - t) + by*t for i, t, w in i_lst], 'o', color="tab:orange"
)
plt.grid()
plt.axis("equal")

plt.plot(left_gon.x_arr, left_gon.y_arr, color="tab:red", alpha=0.5)
plt.plot(l_gon.x_arr, l_gon.y_arr, 'o--', color="tab:red")

plt.plot(right_gon.x_arr, right_gon.y_arr, color="tab:green", alpha=0.5)
plt.plot(r_gon.x_arr, r_gon.y_arr, 'o--', color="tab:green")

plt.show()