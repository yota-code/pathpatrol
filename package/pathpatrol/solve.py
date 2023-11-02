#!/usr/bin/env python3

import math
import sys

import numpy as np
import matplotlib.pyplot as plt

from cc_pathlib import Path

# on pourrait reprendre Piece pour qu'il soit plus flexible et écrire quelques optimisations de la détection de collision


class RoutePart() :
    def __init__(self, A, B) :
        self.A = A
        self.B = B
        
def side(A, B, M) : où M est une liste ou mieux ! un np.array
	ax, ay = A
	bx, by = B
	mx, my = M
	return ((bx - ax)*(my - ay)) - ((by - ay)*(mx - ax))

        
def find_blocking_pieces(self, A, B) :
    """ A and B are considered free points, self is a layer """
    



