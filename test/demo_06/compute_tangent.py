#!/usr/bin/env python3

import collections
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

from pathpatrol.layer import Layer
from pathpatrol.polygon import Polygon

def local_min(m_arr) :
	for i in range(1, len(m_arr) - 1) :
		if m_arr[i-1] <= m_arr[i] >= m_arr[i+1] :
			yield i

def local_max(m_arr) :
	for i in range(1, len(m_arr) - 1) :
		if m_arr[i-1] >= m_arr[i] <= m_arr[i+1] :
			yield i

def local_extremum(m_arr) :
	yield from local_min(m_arr)
	yield from local_max(m_arr)

def local_extrusion(p_gon) :
	for i in range(1, len(p_gon) - 1) :
		(ax, ay), (bx, by), (cx, cy) = p_gon[i-1], p_gon[i], p_gon[i+1]
		if 0 < (bx - ax)*(cy-ay) - (by - ay)*(cx - ax) :
			yield i

def is_freepath(h_gon, a, b) :
	(ax, ay), (bx, by) = h_gon[a], h_gon[b]
	cx, cy = h_gon[0]
	for i in range(1, len(h_gon)) :
		dx, dy = h_gon[i]
		ux, uy = dx - cx, dy - dx
		ma = (ax - cx) * uy - (ay - cy) * ux
		mb = (bx - cx) * uy - (by - cy) * ux
		cx, cy = dx, dy

pth = Path("map.png")
u = Layer().load_rastermap(pth)[0]

e_gon = Polygon(p_arr=u.orig[12:846+1])

"""
k_lst / k_arr recense les indices des protrusions locales,
il n'y a pas besoin de chercher de tangente en dehors de ces points
"""
k_lst = [i for i in local_extrusion(e_gon)]
k_arr = np.array(k_lst)


"""
m_lst / m_arr is the array of all angles detected as candidates,
we should remove the ones we know goes through walls
"""
m_lst = list()
for i, k in enumerate(k_arr) :
	ax, ay = e_gon[int(k)]
	m_arr = np.arctan2(e_gon.y_arr[k_arr] - ay, e_gon.x_arr[k_arr] - ax)
	w_arr = np.unwrap(np.hstack((m_arr[i+1:], m_arr[:i])))
	m_lst.append(w_arr)
m_arr = np.array(m_lst)



v_map = collections.defaultdict(int)

for rk in range(1, len(k_arr) - 1) :
	print("rk", rk)
	r = int(k_arr[rk])

	(ax, ay), (bx, by), (cx, cy) = e_gon[r-1], e_gon[r], e_gon[r+1]
	fa = math.atan2(ay - by, ax - bx)
	fc = math.atan2(cy - by, cx - bx)

	c_lst = list()
	for ck in local_extremum(m_arr[rk,:]) :
		w = (rk+ck+1) % len(k_arr)
		if w == 0 or w == len(k_arr) - 1 :
			# le premier et dernier point de k_arr sont ceux de l'ouverture de la cavité
			continue
		z = int(k_arr[w])

		m = m_arr[rk,ck] % math.tau
		if 0.0 < ((m - fa) % math.tau) < ((fc - fa) % math.tau) :
			# we need to detect if it goes directly through wall
			c_lst.append(ck)
			v = (min(r, k_arr[w]), max(r, k_arr[w]))
			v_map[v] += 1

v_set = set(k for k, v in v_map.items() if v >= 2)

print(v_set)
# print(v_map)


plt.figure()

# k_lst = list(k_arr)
# for i in range(len(e_gon)) :
# 	if i in k_lst :
# 		plt.text(e_gon.x_arr[i] + 0.1, e_gon.y_arr[i] + 0.1, f"{i} ({k_lst.index(i)})", color="tab:green")
# 	else :
# 		plt.text(e_gon.x_arr[i] + 0.1, e_gon.y_arr[i] + 0.1, f"{i}", color="tab:blue")


plt.plot(e_gon.x_arr, e_gon.y_arr, '+--')
plt.axis('equal')
plt.plot(e_gon.x_arr[k_arr], e_gon.y_arr[k_arr], 'P', color="tab:green")
plt.grid()
print(len(v_set))
for a, b in v_set :
	plt.plot(
		[e_gon.x_arr[a], e_gon.x_arr[b]],
		[e_gon.y_arr[a], e_gon.y_arr[b]], alpha=0.5, color="tab:pink"
	)

plt.show()
sys.exit(0)


(ax, ay), (bx, by), (cx, cy) = e_gon[r-1], e_gon[r], e_gon[r+1]
print((ax, ay), (bx, by), (cx, cy))
fa = math.atan2(ay - by, ax - bx)
fc = math.atan2(cy - by, cx - bx)

print(math.degrees(fa) % 360.0, math.degrees(fc) % 360.0, math.degrees(fc - fa) % 360.0)

print(c_lst)

for ck in c_lst :
	z = int(k_arr[(rk+ck+1) % len(k_arr)])

	m = m_arr[rk,ck] % math.tau
	# print(math.degrees(fa) % 360.0, math.degrees(fc) % 360.0, math.degrees(m) % 360.0)
	color = "tab:green" if ((m - fa) % math.tau) < ((fc - fa) % math.tau) else "tab:red"
	plt.plot(
		[e_gon.x_arr[r], e_gon.x_arr[r] + 256*math.cos(m)],
		[e_gon.y_arr[r], e_gon.y_arr[r] + 256*math.sin(m)],
		color=color, alpha=0.5
	)
	plt.plot([e_gon.x_arr[z],], [e_gon.y_arr[z],], '^', color="tab:orange")
plt.grid()

plt.show()

