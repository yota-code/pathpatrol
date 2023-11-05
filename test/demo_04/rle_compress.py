#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

if __name__ == '__main__' :
	pth = Path(sys.argv[1]).resolve()

	lvl = plt.imread(pth)[:,:,0].astype(np.uint16).ravel()

	s = list()
	q = lvl[0]
	c = 0
	for p in lvl :
		if p != q :
			s.append(('b' if q else 'w') + str(c))
			c = 0
		c += 1
		q = p

	pth.with_suffix('.rle').write_text(''.join(s))


