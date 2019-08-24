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
        """Set shape topology"""
        if description is None:
            for i in range(np.shape(self.grid)[0]):
                for j in range(np.shape(self.grid)[1]):
                    self.grid[i, j] = Block(color=self.color)
        else:
            for pos in description:
                self.grid[pos] = Block(color=self.color)

    def set_pivot(self, pivot_pos: tuple):
        """Set Pivot inside Shape.grid

        Parameters
        ----------
        pivot_pos: tuple with pivot coordinates in Shape.grid coordinates
        """
        self.grid[pivot_pos].is_pivot = True
        self.pivot = Point(*pivot_pos)

    @property
    def gameboard_pivot_position(self):
        """Get pivot position in Game.gameboard coordinates"""
        return self.grid[self.pivot].gameboard_position

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

    def prepare_shape_for_rotation(self, sense='cw'):
        """Update all-related grid data for a rotated Shape\n

        Parameters
        ----------
        sense: 'cw' -> clockwise, 'ccw' -> counter-clockwise
        """
        k = 3
        if sense == 'ccw':
            k = 1
        self.width, self.height = self.height, self.width
        old_pivot_gameboard_position = self.gameboard_pivot_position
        self.grid = np.rot90(self.grid, k=k)
        for lin in range(self.width):
            for col in range(self.height):
                if isinstance(self.grid[lin, col], Block):
                    if self.grid[lin, col].is_pivot:
                        self.set_pivot(Point(lin, col))
                        self.grid[lin, col].gameboard_position = \
                            old_pivot_gameboard_position

    def shape_copy(self):
        copy = self.__new__(self.__class__)
        copy.__dict__.update(self.__dict__)
        return copy

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
        self.kys2move = {
            'a': self.move_curr_shape_right,
            'b': self.move_curr_shape_left,
            's': self.move_curr_shape_down,
            'r': self.rotate_curr_shape_clockwise,
        }
        self._get_back_to_shape_generation = False
        self.level = 1
        self.levels = {1: 2, 2: 0.25, 3: 0.125}  # todo response time varying with level
        self.initial_points = {  # todo generalize this for width, heigth != 10, 10
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

    @property
    def sleep_time(self):
        return 0.1 / self.level

    def generate_next_shape(self):
        self.curr_shape = self.sg.__next__()

    @staticmethod
    def fake_gameboard_points_and_shape_points(insertion_point: Point, shape: Shape):
        pivot, grid = shape.pivot, shape.grid
        for lin in range(shape.width):
            for col in range(shape.height):
                if isinstance(grid[lin, col], Block):
                    distx = lin - pivot.x
                    disty = col - pivot.y
                    gb = Point(insertion_point.x + distx, insertion_point.y + disty)
                    yield gb, Point(lin, col)

    def gameboard_points_and_shape_points(self, insertion_point: Point):
        pivot, grid = self.curr_shape.pivot, self.curr_shape.grid
        for lin in range(self.curr_shape.width):
            for col in range(self.curr_shape.height):
                if isinstance(grid[lin, col], Block):
                    distx = lin - pivot.x
                    disty = col - pivot.y
                    gb = Point(insertion_point.x + distx, insertion_point.y + disty)
                    yield gb, Point(lin, col)

    def clear_curr_shape(self):
        for gb_point, s_point in self.gameboard_points_and_shape_points(self.curr_shape.gameboard_pivot_position):
            if isinstance(self.gameboard[gb_point.x, gb_point.y], Block):
                self.gameboard[gb_point.x, gb_point.y] = 0
                # self.curr_shape.grid[s_point.x, s_point.y].gameboard_position = None

    @property
    def initial_shape_point_for_gameboard(self):
        return self.initial_points[self.curr_shape.shape_id]

    def is_curr_shape_rotable(self, insertion_point: Point):
        # todo Block blocking case
        fake_curr_shape = self.curr_shape.shape_copy()
        fake_curr_shape.prepare_shape_for_rotation()
        for g, _ in self.fake_gameboard_points_and_shape_points(insertion_point, fake_curr_shape):
            if not (0 <= g.x < self.width and g.y < self.height):
                return False
        return True

    def is_curr_shape_insertable(self, intl_pnt: Point):
        for gb_point, _ in self.gameboard_points_and_shape_points(intl_pnt):
            if isinstance(self.gameboard[gb_point.x, gb_point.y], Block):
                return False
        return True

    def put_shape_in_gameboard(self, insertion_point: Point):
        pivot, grid = self.curr_shape.pivot, self.curr_shape.grid
        for gb_point, s_point in self.gameboard_points_and_shape_points(insertion_point):
            if isinstance(grid[s_point.x, s_point.y], Block):
                self.gameboard[gb_point.x, gb_point.y] = grid[s_point.x, s_point.y]  # Game.gameboard
                self.gameboard[gb_point.x, gb_point.y].gameboard_position = gb_point  # Blocks.gameboard_position

    def move_curr_shape_right(self):
        if self.curr_shape.can_shape_move_right(self.gameboard):
            pivot = self.curr_shape.gameboard_pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(pivot.x + 1, pivot.y))

    def move_curr_shape_left(self):
        if self.curr_shape.can_shape_move_left(self.gameboard):
            pivot = self.curr_shape.gameboard_pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(pivot.x - 1, pivot.y))

    def move_curr_shape_down(self):
        if self.curr_shape.can_shape_fall(self.gameboard):
            pivot = self.curr_shape.gameboard_pivot_position
            self.clear_curr_shape()
            self.put_shape_in_gameboard(Point(pivot.x, pivot.y - 1))

    def rotate_curr_shape_clockwise(self):
        insertion_point = self.curr_shape.gameboard_pivot_position
        if self.is_curr_shape_rotable(insertion_point):
            self.clear_curr_shape()  # tirei a parte que apaga as posições do gameboard
            self.curr_shape.prepare_shape_for_rotation(sense='cw')
            self.put_shape_in_gameboard(self.curr_shape.gameboard_pivot_position)

    @asyncio.coroutine
    def right(self):
        while True:
            if keyboard.is_pressed('d'):
                self.move_curr_shape_right()
            yield from asyncio.sleep(self.sleep_time)

    @asyncio.coroutine
    def left(self, k=1):
        while True:
            if keyboard.is_pressed('a'):
                self.move_curr_shape_left()
            yield from asyncio.sleep(self.sleep_time)

    @asyncio.coroutine
    def down(self):
        while True:
            if keyboard.is_pressed('s'):
                self.move_curr_shape_down()
            yield from asyncio.sleep(self.sleep_time)

    @asyncio.coroutine
    def rotate(self):
        while True:
            if keyboard.is_pressed('r'):  # todo rotation function should work in extremities as well
                self.rotate_curr_shape_clockwise()
            yield from asyncio.sleep(self.sleep_time)

    @asyncio.coroutine
    def trigger_timer(self):
        self.move_curr_shape_down()  # fall shape one time (natural falling)
        yield from asyncio.sleep(self.levels[self.level])

    @asyncio.coroutine
    def inline_rendering(self):
        while True:
            print(self)
            yield from asyncio.sleep(0)

    def __loop(self):
        futures = [
            self.right(),
            self.left(),
            self.down(),
            self.rotate(),
            self.trigger_timer(),
            self.inline_rendering()
        ]
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
        for position in row:
            if not isinstance(position, Block):
                return False
        return True

    def score_and_remove_row(self, j):
        # todo score_and_remove_row
        self.gameboard[:, j] = 0
        raise Exception()

    def after_curr_shape_cannot_fall(self):
        if self.curr_shape.is_shape_on_gameboard_top(self.height):
            self.game_over()
        else:
            rows = [self.gameboard[:, j] for j in range(self.height)]
            for j, row in enumerate(rows):
                if self.is_row_completed(row):
                    self.score_and_remove_row(j)

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
                continue
            else:
                self.after_curr_shape_cannot_fall()
        print('FINALE GAME OVER!')


class Render(Game):
    def __init__(self):
        super().__init__()


def main():
    Render().start()
