#!/usr/bin/env python3

import enum

class PointTyp(enum.IntEnum):
	UNKNOWN = 0
	OUTSIDE = 1
	CRATER = 2
	CAVITY = 3
	VERTEX = 4
	EDGE = 5
	IMPOSSIBLE = 6
