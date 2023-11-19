#!/usr/bin/env python3

import enum

class PointTyp(enum.IntEnum):
	UNKNOWN = 0
	OUTSIDE = 1 # outside the convex hull, BS < π
	CRATER = 2 # in-between the convex hull and the real shape, BS < τ
	CAVITY = 3 # same, but with no direct access to the outside, τ <= BS
	VERTEX = 4
	EDGE = 5
	IMPOSSIBLE = 6 # inside an obstacle

