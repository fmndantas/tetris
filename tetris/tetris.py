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
        """Pivot inside Shape.grid"""
        self.grid[pivot_pos].is_pivot = True
        self.pivot = Point(*pivot_pos)

    @property
    def pivot_position(self):
        """Pivot position in Game.gameboard"""
        return self.grid[self.pivot].gameboard_position

    def rot90_fall_blck(self, k=1):
        self.grid = np.rot90(self.grid, k=k)

    def can_shape_fall(self, gameboard):
        for lin in range(self.left, self.right + 1):
            if isinstance(gameboard[lin, self.bottom], Block):
                if isinstance(gameboard[lin, self.bottom - 1], Block) or self.bottom == 0:
                    return False
        return True

    def can_shape_move_left(self, gameboard):
        for col in range(self.bottom, self.top + 1):
            if isinstance(gameboard[self.left, col], Block):
                if self.left == 0 or isinstance(gameboard[self.left - 1, col], Block):
                    return False
        return True

    def can_shape_move_right(self, gameboard):
        for col in range(self.bottom, self.top + 1):
            if isinstance(gameboard[self.right, col], Block):
                if self.right == np.shape(gameboard)[0] - 1 or isinstance(gameboard[self.right + 1, col], Block):
                    return False
        return True

    def is_shape_on_gameboard_top(self, height):
        return self.top == height - 1

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
        return str(np.rot90(self.grid)).replace('[', ' ').replace(']', ' ').replace('0', ' ')

    def __str__(self):
        return str(np.rot90(self.grid)).replace('[', ' ').replace(']', ' ').replace('0', ' ')


class ShapeGenerator:
    def __init__(self):
        shape_1 = Shape(4, 1, shape_id=1)  # rect
        shape_1.set_pivot((2, 0))

        shape_2 = Shape(2, 3, description=((0, 0), (0, 1), (0, 2), (1, 0)), shape_id=2)  # ell
        shape_2.set_pivot((0, 1))

        shape_3 = Shape(2, 3, description=((0, 0), (1, 0), (1, 1), (1, 2)), shape_id=3)  # mirrored ell
        shape_3.set_pivot((1, 1))

        shape_4 = Shape(shape_id=4)  # square
        shape_4.set_pivot((1, 1))

        shape_5 = Shape(3, 2, description=((0, 0), (1, 0), (1, 1), (2, 1)), shape_id=5)  # unfinished 's'
        shape_5.set_pivot((1, 0))

        shape_6 = Shape(3, 2, description=((0, 1), (1, 1), (1, 0), (2, 0)), shape_id=6)  # unfinished 'z'
        shape_6.set_pivot((1, 0))

        shape_7 = Shape(3, 2, description=((0, 0), (1, 0), (2, 0), (1, 1)), shape_id=7)  # symmetrical ell
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
        self.shape_generator = ShapeGenerator()
        self.sg = self.shape_generator()
        self.is_game_over = False
        self.keys = {'right': 'd', 'left': 'a', 'down': 's', 'rotate': 'r'}
        self._get_back_to_shape_generation = False
        self.level = 1
        self.levels = {1: 1, 2: 0.5, 3: 0.25}
        self.initial_points = {
            # todo generalize this for width, heigth != 10, 10
            1: Point(4, 9),
            2: Point(4, 8),
            3: Point(5, 8),
            4: Point(4, 9),
            5: Point(4, 8),
            6: Point(4, 8),
            7: Point(4, 8)
        }

    def __repr__(self):
        return np.rot90(self.gameboard).__str__().replace('0', '.')

    def __str__(self):
        return np.rot90(self.gameboard).__str__().replace('0', '.')

    def generate_next_shape(self):
        self.curr_shape = self.sg.__next__()

    def gameboard_points_and_shape_points(self, intl_pnt: Point):
        pivot, grid = self.curr_shape.pivot, self.curr_shape.grid
        for lin in range(self.curr_shape.width):
            for col in range(self.curr_shape.height):
                if isinstance(grid[lin, col], Block):
                    distx = lin - pivot.x
                    disty = col - pivot.y
                    gb = Point(intl_pnt.x + distx, intl_pnt.y + disty)
                    yield gb, Point(lin, col)

    def clear_curr_shape(self):
        for gb_point, s_point in self.gameboard_points_and_shape_points(self.curr_shape.pivot_position):
            if isinstance(self.gameboard[gb_point.x, gb_point.y], Block):
                self.gameboard[gb_point.x, gb_point.y] = 0
                self.curr_shape.grid[s_point.x, s_point.y].gameboard_position = None

    @property
    def initial_shape_point_for_gameboard(self):
        return self.initial_points[self.curr_shape.shape_id]

    def is_curr_shape_insertable(self, intl_pnt: Point):
        for point, _ in self.gameboard_points_and_shape_points(intl_pnt):
            if isinstance(self.gameboard[point.x, point.y], Block):
                return False
        return True

    def put_shape_in_gameboard(self, intl_pnt: Point, shape=None):
        if shape is not None:
            self.curr_shape = shape
        pivot, grid = self.curr_shape.pivot, self.curr_shape.grid
        for gb_point, s_point in self.gameboard_points_and_shape_points(intl_pnt):
            if isinstance(grid[s_point.x, s_point.y], Block):
                self.gameboard[gb_point.x, gb_point.y] = grid[s_point.x, s_point.y]
                self.gameboard[gb_point.x, gb_point.y].gameboard_position = gb_point  # Blocks.gameboard_position

    def move_curr_shape_right(self, k=1):
        if self.curr_shape.can_shape_move_right(self.gameboard):
            curr_pivot_pos = self.curr_shape.pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(curr_pivot_pos.x + 1, curr_pivot_pos.y))

    def move_curr_shape_left(self, k=1):
        if self.curr_shape.can_shape_move_left(self.gameboard):
            curr_pivot_pos = self.curr_shape.pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(curr_pivot_pos.x - 1, curr_pivot_pos.y))

    def move_curr_shape_down(self, k=1):
        if self.curr_shape.can_shape_fall(self.gameboard):
            curr_pivot_pos = self.curr_shape.pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(curr_pivot_pos.x, curr_pivot_pos.y - 1))

    def rotate_curr_shape_clockwise(self, k=3):
        self.curr_shape = np.rot90(self.curr_shape, k=k)

    @asyncio.coroutine
    def right(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['right']):
                self.move_curr_shape_right(k=k)
            yield from asyncio.sleep(0.1)  # todo change the time to a class constant

    @asyncio.coroutine
    def left(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['left']):
                self.move_curr_shape_left(k=k)
            yield from asyncio.sleep(0.1)

    @asyncio.coroutine
    def down(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['down']):
                self.move_curr_shape_down(k=k)
            yield from asyncio.sleep(0.1)

    @asyncio.coroutine
    def rotate(self, k=1):
        while 1:
            if keyboard.is_pressed(self.keys['rotate']):  # todo rotation function should work in extremities as well
                self.rotate_curr_shape_clockwise(k=k)
            yield from asyncio.sleep(0.1)

    @asyncio.coroutine
    def trigger_timer(self):
        self.move_curr_shape_down()  # fall shape one time (natural falling)
        yield from asyncio.sleep(self.levels[self.level])

    @asyncio.coroutine
    def __loop(self):
        futures = [self.right(),
                   self.left(),
                   self.down(),
                   self.rotate(),
                   self.trigger_timer()]
        done, pendent = yield from asyncio.wait(futures, return_when=FIRST_COMPLETED)
        for future in pendent:
            future.cancel()
        if not self.curr_shape.can_shape_fall(self.gameboard):
            self._get_back_to_shape_generation = True

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
        pass

    def after_curr_shape_cannot_fall(self):
        if self.curr_shape.is_shape_on_gameboard_top(self.height):
            self.game_over()
        else:
            rows = [self.gameboard[:, j] for j in range(self.height)]
            for height, row in enumerate(rows):
                if self.is_row_completed(row):
                    self.score_and_remove_row(height)

    def start(self):
        while not self.is_game_over:
            self._get_back_to_shape_generation = False
            self.generate_next_shape()
            intl_pnt = self.initial_shape_point_for_gameboard
            if self.is_curr_shape_insertable(intl_pnt):
                self.put_shape_in_gameboard(intl_pnt)
            else:
                self.game_over()
                continue
            if self.curr_shape.can_shape_fall(self.gameboard):
                while not self._get_back_to_shape_generation:
                    self.__trigger_loop()  # fall shape one time and trigger loop N ms
                    print(self)  # todo clear this after debug
                continue
            else:
                self.after_curr_shape_cannot_fall()
        print('FINALE GAME OVER!')
