#!/usr/bin/env python3

import numpy as np


def angle_3pt(A, B, M) :
	""" return the angle between AB and AM """
	(ax, ay), (bx, by), (mx, my) = A, B, M
	
	ABx = bx - ax
	ABy = by - ay
	AMx = mx - ax
	AMy = my - ay

	v = (ABx * AMy) - (ABy * AMx)
	ABn = np.sqrt(ABx**2 + ABy**2)
	AMn = np.sqrt(AMx**2 + AMy**2)
	
	return np.arcsin(v / (ABn * AMn))


if __name__ == '__main__' :
	print(np.degrees(angle_3pt((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))))