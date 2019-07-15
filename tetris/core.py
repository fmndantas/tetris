import numpy as np
from numpy.random import choice

class Block(object):
    def __init__(self, height=2, width=2, description=None, color=None):
        self.height = height
        self.width = width
        self._grid = np.ones((self.height, self.width)) * -1
        self.set_shape(description)
        if color is not None:
            self.color = color

    def set_shape(self, description):
        if description is None:
            self._grid[...] = 0
        else:
            for pos in description:
                self._grid[pos] = 0

    def __str__(self):
        return self._grid.__str__()


class GameBoard(object):

    BLOCK_1 = Block(1, 4)                                    # rect
    BLOCK_2 = Block(2, 3, ((0, 0), (0, 1), (0, 2), (1, 0)))  # mirrored ell
    BLOCK_3 = Block(2, 3, ((1, 0), (1, 1), (1, 2), (0, 2)))  # ell
    BLOCK_4 = Block(2, 2)                                    # square
    BLOCK_5 = Block(2, 3, ((1, 0), (1, 1), (0, 1), (0, 2)))  # unfinished 's'
    BLOCK_6 = Block(2, 3, ((0, 0), (0, 1), (1, 1), (1, 2)))  # unfinished 'z'
    BLOCK_7 = Block(2, 3, ((1, 0), (1, 1), (0, 1), (1, 2)))
    BLOCKS = [BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4, BLOCK_5, BLOCK_6, BLOCK_7]

    def __init__(self, width=25, height=25):
        self.width = width
        self.height = height
        self.speed = 500  # ms
        self._game_board = np.ones((self.height, self.width)) * -1
        self._falling_block = None
        self._bg = self.block_generator()

    def block_generator(self):
        while True:
            yield choice(self.BLOCKS)