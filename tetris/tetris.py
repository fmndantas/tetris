import numpy as np
from numpy.random import choice
from collections import namedtuple

Point = namedtuple('Point', 'x y')


class Block(object):
    def __init__(self, color='w', is_pivot=False):
        self.color = color
        self.is_pivot = is_pivot
        self.gameboard_position = None

    def __repr__(self):
        if self.is_pivot:
            return u"\u25A0"
        return u"\u25A1"

    @property
    def gb_position(self):
        return self.gameboard_position

    @gb_position.setter
    def gb_position(self, gb_pos):
        self.gameboard_position = gb_pos


class Shape(object):
    def __init__(self, height=2, width=2, description=None, color='w', shape_id=None):
        self.height = height
        self.width = width
        self.grid = np.zeros((self.height, self.width), object)
        self.pivot = None
        self.color = color
        self.set_shape(description)
        self.shape_id = shape_id

    def set_shape(self, description):
        if description is None:
            for i in range(np.shape(self.grid)[0]):
                for j in range(np.shape(self.grid)[1]):
                    self.grid[i, j] = Block(color=self.color)
        else:
            for pos in description:
                self.grid[pos] = Block(color=self.color)

    def set_pivot(self, pivot_pos):
        self.grid[pivot_pos].is_pivot = True
        self.pivot = Point(*pivot_pos)

    def rot90_fall_blck(self, k=1):
        self.grid = np.rot90(self.grid, k=k)

    def __repr__(self):
        return str(self.grid).replace('[', ' ').replace(']', ' ').replace('0', ' ')

    def __str__(self):
        return str(self.grid).replace('[', ' ').replace(']', ' ').replace('0', ' ')


class ShapeGenerator:
    def __init__(self):
        shape_1 = Shape(1, 4, shape_id=1)                                     # rect - revised
        shape_1.set_pivot((0, 2))

        shape_2 = Shape(3, 2, ((0, 0), (1, 0), (2, 0), (2, 1)), shape_id=2)   # ell - revised
        shape_2.set_pivot((1, 0))

        shape_3 = Shape(3, 2, ((0, 1), (1, 1), (2, 1), (2, 0)), shape_id=3)   # mirrored ell - revised
        shape_3.set_pivot((1, 1))

        shape_4 = Shape(2, 2, shape_id=4)                                     # square - revised
        # no pivot

        shape_5 = Shape(2, 3, ((1, 0), (1, 1), (0, 1), (0, 2)), shape_id=5)   # unfinished 's'
        shape_5.set_pivot((1, 1))

        shape_6 = Shape(2, 3, ((0, 0), (0, 1), (1, 1), (1, 2)), shape_id=6)   # unfinished 'z'
        shape_6.set_pivot((1, 1))

        shape_7 = Shape(2, 3, ((1, 0), (1, 1), (0, 1), (1, 2)), shape_id=7)  # symmetrical ell
        shape_7.set_pivot((1, 1))

        self.shapes = [shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, shape_7]
        for shape in self.shapes: shape.rot90_fall_blck(2)

    def __call__(self, *args, **kwargs):
        return self.shape_generator()

    def shape_generator(self):
        while True:
            yield choice(self.shapes)


class GameBoard(object):
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.game_board = np.zeros((height, width), object)
        self.falling_shape = None
        self.sg = ShapeGenerator()()
        self.pivot_initial_pos = {0: (0, 10)}

    def generate_next_shape(self):
        self.falling_shape = self.sg.__next__()

    def put_shape_in_gameboard(self, intl_pnt: Point) -> None:
        pivot = self.falling_shape.pivot
        grid = self.falling_shape.grid

    def can_block_fall(self):
        pass

    def fall_block(self):
        pass

    def loop(self):
        pass