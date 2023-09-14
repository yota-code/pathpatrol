#!/usr/bin/env python3

import collections
import math

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

def open_hgt(pth) :
	siz = pth.stat().st_size
	dim = int(math.sqrt(siz/2))
	
	assert dim*dim*2 == siz, 'Invalid file size'
	
	data = np.fromfile(pth, np.dtype('>i2'), dim*dim).reshape((dim, dim))

	print(data.shape)

	return data

lvl = open_hgt(Path('srtm/N44E002.hgt')) > 550

row, col = lvl.shape

def color(lvl, r, c, n) :
	q_lst = set()
	d_set = set()
	
	o = lvl[r, c]
	for i in [-1, 1] :
		for j in [-1, 1] :
			if 0 <= r+i < row and 0 <= c+j < col :
				k = (r+i, c+j)
				if lvl[k] == o :
					q_lst.add(k)

n = 1
for r in range(row) :
	for c in range(col) :
		if lvl[r, c] < 2 :
			pass


plt.imshow(lvl)
plt.show()