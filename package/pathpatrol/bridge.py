#!/usr/bin/env python3

import itertools
import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

def find_crosslines(a, b) :
	p = (None, None, None, None)

	bi, bj = b.tangent(a[0])
	ai, _ = a.tangent(b[bi])
	_, aj = a.tangent(b[bj])

	while (ai, aj, bi, bj) != p :
		# plt.figure()
		# a.plot()
		# b.plot()
		# plt.plot([a[ai][0], b[bi][0]], [a[ai][1], b[bi][1]])
		# plt.plot([a[aj][0], b[bj][0]], [a[aj][1], b[bj][1]])
		# plt.grid()
		# plt.show()

		p = (ai, aj, bi, bj)

		bi, _ = b.tangent(a[ai])
		_, bj = b.tangent(a[aj])
		ai, _ = a.tangent(b[bi])
		_, aj = a.tangent(b[bj])

	return ai, aj, bi, bj

def find_outlines(a, b) :
	p = (None, None, None, None)

	bj, bi = b.tangent(a[0])
	ai, _ = a.tangent(b[bi])
	_, aj = a.tangent(b[bj])
	
	while (ai, aj, bi, bj) != p :
		# plt.figure()
		# a.plot()
		# b.plot()
		# plt.title(str([ai, aj, bi, bj]))
		# plt.plot([a[ai][0], b[bi][0]], [a[ai][1], b[bi][1]])
		# plt.plot([a[aj][0], b[bj][0]], [a[aj][1], b[bj][1]])
		# plt.grid()
		# plt.show()

		p = (ai, aj, bi, bj)

		bj, _ = b.tangent(a[aj])
		_, bi = b.tangent(a[ai])
		ai, _ = a.tangent(b[bi])
		_, aj = a.tangent(b[bj])

	return ai, aj, bi, bj