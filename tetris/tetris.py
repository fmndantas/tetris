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

    @property
    def pivot_position(self):
        return self.grid[self.pivot].gameboard_position

    def rot90_fall_blck(self, k=1):
        self.grid = np.rot90(self.grid, k=k)

    def can_shape_fall(self, gameboard):
        for lin in range(self.left, self.right):
            if isinstance(gameboard[lin, self.bottom - 1], Block):
                return False
            continue
        return True

    def can_shape_left(self):
        print('can_shape_left')
        return self.left > 0

    def can_shape_right(self, width):
        print('can_shape_right')
        return self.right < width - 1

    def is_shape_on_gameboard_top(self, height):
        print('is_shape_on_gameboard_top')
        return self.top >= height - 1  # todo check this

    @property
    def bottom(self):
        return min(self.grid[lin, col].gameboard_position.y
                   for lin in range(self.width)
                   for col in range(self.height)
                   if isinstance(self.grid[lin, col], Block))

    @property
    def top(self):
        return max(self.grid[lin, col].gameboard_position.y
                   for lin in range(self.width)
                   for col in range(self.height)
                   if isinstance(self.grid[lin, col], Block))

    @property
    def left(self):
        return min(self.grid[lin, col].gameboard_position.x
                   for lin in range(self.width)
                   for col in range(self.height)
                   if isinstance(self.grid[lin, col], Block))

    @property
    def right(self):
        return max(self.grid[lin, col].gameboard_position.x
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


class Game(object):
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.gameboard = np.zeros((width, height), object)
        self.curr_shape = None
        self.sg = ShapeGenerator()()  # todo this is ugly
        self.pivot_initial_pos = {0: (0, 10)}
        self.is_game_over = False
        self.level = (i + 1 for i in range(20))
        self.keys = {'right': 'd', 'left': 'a', 'down': 's', 'rotate': 'r'}
        self._get_back_to_shape_generation = False
        self.levels = {1: 5}

    def generate_next_shape(self):
        self.curr_shape = self.sg.__next__()

    def clear_curr_shape(self):
        for lin in range(self.curr_shape.width):
            for col in range(self.curr_shape.height):
                if isinstance(self.curr_shape.grid[lin, col], Block):
                    self.gameboard[self.curr_shape.grid[lin, col].gameboard_position] = 0
                    self.curr_shape.grid[lin, col].gameboard_position = None

    def put_shape_in_gameboard(self, intl_pnt: Point, shape=None) -> None:
        if shape is not None:
            self.curr_shape = shape
        pivot, grid = self.curr_shape.pivot, self.curr_shape.grid
        for lin in range(self.curr_shape.width):
            for col in range(self.curr_shape.height):
                if isinstance(grid[lin][col], Block):
                    distx = lin - pivot.x
                    disty = col - pivot.y
                    gb = Point(intl_pnt.x + distx, intl_pnt.y + disty)
                    self.gameboard[gb.x, gb.y] = grid[lin, col]
                    self.gameboard[gb.x, gb.y].gameboard_position = gb  # changing Blocks.gameboard_position

    def move_curr_shape_right(self, k=1):
        if self.curr_shape.can_shape_right(self.width):
            curr_pivot_pos = self.curr_shape.pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(curr_pivot_pos.x + 1, curr_pivot_pos.y))

    def move_curr_shape_left(self, k=1):
        if self.curr_shape.can_shape_left:
            curr_pivot_pos = self.curr_shape.pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(curr_pivot_pos.x - 1, curr_pivot_pos.y))

    def move_curr_shape_down(self, k=1):
        print('move_current_shape_down')

    def rotate_curr_shape_clockwise(self, k=1):
        print('rotate_current_shape_clockwise')

    @asyncio.coroutine
    def right(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['right']) \
                    and self.curr_shape.can_shape_right(self.gameboard):
                self.move_curr_shape_right(k=k)
            yield from asyncio.sleep(0)

    @asyncio.coroutine
    def left(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['left']) \
                    and self.curr_shape.can_shape_left():
                self.move_curr_shape_left(k=k)
            yield from asyncio.sleep(0)

    @asyncio.coroutine
    def down(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['down']) \
                    and self.curr_shape.can_shape_fall(self.gameboard):
                self.move_curr_shape_down(k=k)
            yield from asyncio.sleep(0)

    @asyncio.coroutine
    def rotate(self, k=1):
        while 1:
            if keyboard.is_pressed(
                    self.keys['rotate']):  # todo rotation function should work in extremities as well
                self.rotate_curr_shape_clockwise(k=k)
            yield from asyncio.sleep(0)

    @asyncio.coroutine
    def trigger_timer(self):
        self.move_curr_shape_down()  # fall shape one time (natural falling)
        yield from asyncio.sleep(self.levels[self.level])

    @asyncio.coroutine
    def asynchronous_can_shape_fall(self):
        while 1:
            if not self.curr_shape.can_shape_fall(self.gameboard):
                self.after_curr_shape_cannot_fall()  # after asynchronous_can_shape_fall return
                self._get_back_to_shape_generation = True
                return None
            yield from asyncio.sleep(0)

    @asyncio.coroutine
    def __loop(self):
        futures = [self.right(),
                   self.left(),
                   self.down(),
                   self.rotate(),
                   self.trigger_timer(),
                   self.asynchronous_can_shape_fall()]
        done, pendent = yield from asyncio.wait(futures, return_when=FIRST_COMPLETED)
        for future in pendent:
            future.cancel()

    def __trigger_loop(self):
        ioloop = asyncio.new_event_loop()
        ioloop.run_until_complete(self.__loop())
        ioloop.close()

    def game_over(self):
        self.is_game_over = True

    @staticmethod
    def is_row_completed(row):
        for item in row:
            if not isinstance(item, Block):
                return False
        return True

    def score_and_remove_row(self, height):
        print('score_and_remove_row')

    def after_curr_shape_cannot_fall(self):
        if self.curr_shape.is_shape_on_gameboard_top:
            self.game_over()
        else:
            rows = [self.gameboard[:, j] for j in range(self.height)]
            for height, row in enumerate(rows):
                if self.is_row_completed(row):
                    self.score_and_remove_row(height)

    def start(self):
        while not self.is_game_over:
            self.generate_next_shape()
            self.put_shape_in_gameboard(Point(5, 5))  # todo start points?
            if self.curr_shape.can_shape_fall(self.gameboard):
                while not self._get_back_to_shape_generation:
                    self.__trigger_loop()  # fall shape one time and trigger loop N ms
                continue
            else:
                self.after_curr_shape_cannot_fall()
        print('game over')
