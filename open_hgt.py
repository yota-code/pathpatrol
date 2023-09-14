#!/usr/bin/env python3

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

N44E001_arr = open_hgt(Path('srtm/N44E001.hgt'))
N43E001_arr = open_hgt(Path('srtm/N43E001.hgt'))
N44E002_arr = open_hgt(Path('srtm/N44E002.hgt'))
N43E002_arr = open_hgt(Path('srtm/N43E002.hgt'))

puycelci = np.hstack((np.vstack((N44E001_arr, N43E001_arr)), np.vstack((N44E002_arr, N43E002_arr))))

puycelci = N44E002_arr

N44E006_arr = open_hgt(Path('srtm/N44E006.hgt'))
N43E006_arr = open_hgt(Path('srtm/N43E006.hgt'))
N44E007_arr = open_hgt(Path('srtm/N44E007.hgt'))
N43E007_arr = open_hgt(Path('srtm/N43E007.hgt'))

roya = np.hstack((np.vstack((N44E006_arr, N43E006_arr)), np.vstack((N44E007_arr, N43E007_arr))))


# import matplotlib.image

# for i in range(0, 5000, 50) :
# 	print(i)
# 	matplotlib.image.imsave(f'puycelci/{i:04d}.png', (puycelci > i))
# 	if np.sum((puycelci > i)) == 0.0 :
# 		break

with Path("puycelci_500.npz").open('wb') as fid :
	np.savez(fid, data=(puycelci > 500))

plt.imshow(np.floor(puycelci / 50) * 50 > 500)
plt.show()