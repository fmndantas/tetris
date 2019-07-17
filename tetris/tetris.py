import asyncio
from collections import namedtuple
from concurrent.futures import FIRST_COMPLETED

import keyboard
import numpy as np
from numpy.random import choice

Point = namedtuple('Point', ('x', 'y'))


class Block(object):
    def __init__(self, color='w', is_pivot=False):
        self.color = color
        self.is_pivot = is_pivot
        self.gameboard_position = None

    def __repr__(self):
        if self.is_pivot:
            return u"\u25A0"
        return u"\u25A1"


class Shape(object):
    def __init__(self, width=2, height=2, description=None, color='w', shape_id=None):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.width, self.height), object)
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

    def can_block_fall(self, gameboard):
        for lin in range(self.left, self.right):
            if isinstance(gameboard[lin, self.bottom - 1], Block):
                return False
            continue
        return True

    @property
    def bottom(self):
        return min(self.grid[lin, col].gameboard_position[1]
                   for lin in range(self.width)
                   for col in range(self.height)
                   if isinstance(self.grid[lin, col], Block))

    @property
    def top(self):
        return max(self.grid[lin, col].gameboard_position[1]
                   for lin in range(self.width)
                   for col in range(self.height)
                   if isinstance(self.grid[lin, col], Block))

    @property
    def left(self):
        return min(self.grid[lin, col].gameboard_position[0]
                   for lin in range(self.width)
                   for col in range(self.height)
                   if isinstance(self.grid[lin, col], Block))

    @property
    def right(self):
        return max(self.grid[lin, col].gameboard_position[0]
                   for lin in range(self.width)
                   for col in range(self.height)
                   if isinstance(self.grid[lin, col], Block))

    def __repr__(self):
        return str(self.grid).replace('[', ' ').replace(']', ' ').replace('0', ' ')

    def __str__(self):
        return str(self.grid).replace('[', ' ').replace(']', ' ').replace('0', ' ')


class ShapeGenerator:
    def __init__(self):
        shape_1 = Shape(4, 1, shape_id=1)  # rect
        shape_1.set_pivot((2, 0))

        shape_2 = Shape(2, 3, ((0, 0), (0, 1), (0, 2), (1, 0)), shape_id=2)  # ell
        shape_2.set_pivot((0, 1))

        shape_3 = Shape(2, 3, ((0, 0), (1, 0), (1, 1), (1, 2)), shape_id=3)  # mirrored ell
        shape_3.set_pivot((1, 1))

        shape_4 = Shape(shape_id=4)  # square
        shape_4.set_pivot((1, 1))

        shape_5 = Shape(3, 2, ((0, 0), (1, 0), (1, 1), (2, 1)), shape_id=5)  # unfinished 's'
        shape_5.set_pivot((1, 0))

        shape_6 = Shape(3, 2, ((0, 1), (1, 1), (1, 0), (2, 0)), shape_id=6)  # unfinished 'z'
        shape_6.set_pivot((1, 0))

        shape_7 = Shape(3, 2, ((0, 0), (1, 0), (2, 0), (1, 1)), shape_id=7)  # symmetrical ell
        shape_7.set_pivot((1, 0))

        self.shapes = [shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, shape_7]

    def __call__(self, *args, **kwargs):
        return self.shape_generator()

    def shape_generator(self):
        while True:
            yield choice(self.shapes)


_LEVELS = {1: 5}


class Game(object):
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.gameboard = np.zeros((width, height), object)
        self.falling_shape = None
        self.sg = ShapeGenerator()()
        self.pivot_initial_pos = {0: (0, 10)}
        self.is_game_over = False
        self.level = 1
        self.keys = {'right': 'd', 'left': 'a', 'down': 'd', 'rotate': 'r'}

    def generate_next_shape(self):
        self.falling_shape = self.sg.__next__()

    def put_shape_in_gameboard(self, intl_pnt: Point) -> None:
        pivot = self.falling_shape.pivot
        grid = self.falling_shape.grid
        for lin in range(self.falling_shape.width):
            for col in range(self.falling_shape.height):
                if isinstance(grid[lin][col], Block):
                    distx = lin - pivot.x
                    disty = col - pivot.y
                    gb = Point(intl_pnt.x + distx, intl_pnt.y + disty)
                    self.gameboard[gb.x, gb.y] = grid[lin, col]
                    self.gameboard[gb.x, gb.y].gameboard_position = gb.x, gb.y  # changing Blocks.gameboard_position

    def move_falling_shape_right(self, k=1):
        print('falling shape to right')

    def move_falling_shape_left(self, k=1):
        print('falling shape to left')

    def move_falling_shape_down(self, k=1):
        print('move falling shape down')

    def rotate_falling_shape_clockwise(self, k=1):
        print('rotate falling shape clockwise')

    @asyncio.coroutine
    def right(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['right']):
                self.move_falling_shape_right(k=k)
            yield from asyncio.sleep(0)

    @asyncio.coroutine
    def left(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['left']):
                self.move_falling_shape_left(k=k)
            yield from asyncio.sleep(0)

    @asyncio.coroutine
    def trigger_timer(self):
        yield from asyncio.sleep(_LEVELS[self.level])

    @asyncio.coroutine
    def __inner_loop(self):
        futures = [self.right(), self.left(), self.trigger_timer()]
        done, pendent = yield from asyncio.wait(futures, return_when=FIRST_COMPLETED)
        for future in pendent:
            future.cancel()

    def game_over(self):
        pass

    def start(self):
        while not self.is_game_over:
            self.generate_next_shape()
            self.put_shape_in_gameboard(Point(5, 5))
            ioloop = asyncio.new_event_loop()
            ioloop.run_until_complete(self.__inner_loop())
            ioloop.close()
        self.game_over()
