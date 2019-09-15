import asyncio
from collections import namedtuple
from concurrent.futures import FIRST_COMPLETED
from copy import deepcopy

import keyboard
import numpy as np
from numpy.random import randint

Point = namedtuple('Point', ('x', 'y'))

COLORS = {
    "r": "\u001b[31m",
    "g": "\u001b[32m",
    "y": "\u001b[33m",
    "m": "\u001b[35m",
    "c": "\u001b[36m",
    "w": "\u001b[37m",
    "b": "\u001b[34m"
}


class Block(object):
    def __init__(self, color='w', is_pivot=False):
        self.color = color
        self.is_pivot = is_pivot
        self.gameboard_position = None

    def __repr__(self):
        if self.is_pivot:
            return COLOURS[self.color] + u"\u25A0"
        return COLOURS[self.color] + u"\u25A1"

    def __eq__(self, other):
        if isinstance(other, Block):
            return self.color == other.color \
                   and self.is_pivot == other.is_pivot \
                   and self.gameboard_position == other.gameboard_position


class Shape(object):
    def __init__(self, width=2, height=2, description=None, color='w', shape_id=None):
        self.width = width
        self.height = height
        self.grid = np.zeros((self.width, self.height), object)
        self.pivot = None
        self.color = color
        self.set_shape(description)
        self.shape_id = shape_id

    def __eq__(self, other):
        for i in range(self.width):
            for j in range(self.height):
                if self.grid[i, j] != other.grid[i, j]:
                    return False
        return True

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

    @gameboard_pivot_position.setter
    def gameboard_pivot_position(self, position: Point):
        self.grid[self.pivot.x, self.pivot.y].gameboard_position = position

    @property
    def lower_block_per_col(self):
        for row in range(self.width):
            min_col = np.min(np.nonzero(self.grid[row]))
            yield self.grid[row, min_col].gameboard_position

    def can_shape_fall(self, gameboard):
        for lower in self.lower_block_per_col:
            if isinstance(gameboard[lower.x, lower.y - 1], Block) or self.bottom == 0:
                return False
        return True

    def extremed_block_per_row(self, side):
        for col in range(self.height):
            if side == 'l':
                side_row = np.min(np.nonzero(self.grid[:, col]))
            else:
                side_row = np.max(np.nonzero(self.grid[:, col]))
            yield self.grid[side_row, col].gameboard_position

    def can_shape_move_left(self, gameboard):
        for most_left in self.extremed_block_per_row(side='l'):
            if self.left == 0 or isinstance(gameboard[most_left.x - 1, most_left.y], Block):
                return False
        return True

    def can_shape_move_right(self, gameboard):
        for most_right in self.extremed_block_per_row(side='r'):
            if self.right == np.shape(gameboard)[0] - 1 or isinstance(gameboard[most_right.x + 1, most_right.y], Block):
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
                    else:
                        self.grid[lin, col].gameboard_position = None

    def shape_copy(self):
        return deepcopy(self)

    def __repr__(self):
        return str(np.rot90(self.grid)).replace('[', ' ').replace(']', ' ').replace('0', ' ')

    def __str__(self):
        return str(np.rot90(self.grid)).replace('[', ' ').replace(']', ' ').replace('0', ' ')


class ShapeGenerator:
    def __init__(self):
        shape_1 = Shape(4, 1, shape_id=1, color='r')  # rect
        shape_1.set_pivot((2, 0))

        shape_2 = Shape(2, 3, description=((0, 0), (0, 1), (0, 2), (1, 0)), shape_id=2, color='g')  # ell
        shape_2.set_pivot((0, 1))

        shape_3 = Shape(2, 3, description=((0, 0), (1, 0), (1, 1), (1, 2)), shape_id=3, color='y')  # mirrored ell
        shape_3.set_pivot((1, 1))

        shape_4 = Shape(shape_id=4, color='m')  # square
        shape_4.set_pivot((1, 1))

        shape_5 = Shape(3, 2, description=((0, 0), (1, 0), (1, 1), (2, 1)), shape_id=5, color='c')  # unfinished 's'
        shape_5.set_pivot((1, 0))

        shape_6 = Shape(3, 2, description=((0, 1), (1, 1), (1, 0), (2, 0)), shape_id=6, color='w')  # unfinished 'z'
        shape_6.set_pivot((1, 0))

        shape_7 = Shape(3, 2, description=((0, 0), (1, 0), (2, 0), (1, 1)), shape_id=7, color='b')  # symmetrical ell
        shape_7.set_pivot((1, 0))

        self.shapes = [shape_1, shape_2, shape_3, shape_4, shape_5, shape_6, shape_7]

    def __call__(self, choice):
        return self.shapes[choice].shape_copy()


class Game(object):
    def __init__(self, width=10, height=20):
        self.width = width
        self.height = height
        self.gameboard = np.zeros((width, height), object)
        self.game_score = 0
        self.level = 1
        self.levels = {1: 1, 2: 0.25, 3: 0.125}  # todo response time varying with level
        self.curr_shape = None
        self.shape_generator = ShapeGenerator()
        self.is_game_over = False
        self.kys2move = {
            'a': self.move_curr_shape_right,
            'b': self.move_curr_shape_left,
            's': self.move_curr_shape_down,
            'r': self.rotate_curr_shape_clockwise,
        }
        self._get_back_to_shape_generation = False
        self.initial_points = {
            1: Point(4, self.height - 1),
            2: Point(4, self.height - 2),
            3: Point(5, self.height - 2),
            4: Point(4, self.height - 1),
            5: Point(4, self.height - 2),
            6: Point(4, self.height - 2),
            7: Point(4, self.height - 2)
        }

    def __repr__(self):
        return np.rot90(self.gameboard).__str__().replace('0', '.')

    def __str__(self):
        return np.rot90(self.gameboard).__str__().replace('0', '.')

    def reset(self):
        self.gameboard = np.zeros_like(self.gameboard)

    @property
    def sleep_time(self):
        return 0.05 / self.level

    def generate_next_shape(self, ):
        self.curr_shape = self.shape_generator(randint(0, 7))

    @staticmethod
    def fake_gameboard_points_and_shape_points(insertion_point: Point, shape: Shape):
        """Emulate the emitted points when shape is being inserted on gameboard
        """
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

    @property
    def initial_shape_point_for_gameboard(self):
        """Returns the initial gameboard point based on shape id"""
        return self.initial_points[self.curr_shape.shape_id]

    def gameboard_copy(self, mode):
        """Returns a copy of current gameboard without the most recently version of current_shape
        """
        copy = deepcopy(self.gameboard)
        if mode == "no_current_shape":
            fake_shape = self.curr_shape.shape_copy()
            fake_coords = set([fake_shape.grid[i, j].gameboard_position
                               for i in range(fake_shape.width)
                               for j in range(fake_shape.height)
                               if isinstance(fake_shape.grid[i, j], Block)])
            for i in range(self.width):
                for j in range(self.height):
                    copy[i, j] = 0 if Point(i, j) in fake_coords else copy[i, j]
        return copy

    def analyze_rotation_at_frontiers(self, insertion_point: Point, prepared_fake_shape: Shape):
        for g, _ in self.fake_gameboard_points_and_shape_points(insertion_point, prepared_fake_shape):
            if not (0 <= g.x < self.width and 0 <= g.y < self.height):
                return False
        return True

    def analyze_rotation_at_neighborhood(self, insertion_point: Point, prepared_fake_shape: Shape):
        fake_gameboard = self.gameboard_copy(mode="no_current_shape")
        for g, _ in self.fake_gameboard_points_and_shape_points(insertion_point, prepared_fake_shape):
            if isinstance(fake_gameboard[g.x, g.y], Block):
                return False
        return True

    def frontier_offset(self, insertion_point: Point, shape: Shape):
        """Returns the displacement that is used by curr_shape_offset to
        perform translation after rotation
        """
        for g, _ in self.fake_gameboard_points_and_shape_points(insertion_point, shape):
            if g.x >= self.width - 1:
                return Point(-1, 0)
            elif g.x <= 0:
                return Point(1, 0)
            elif g.y > self.height - 1:
                return Point(0, -1)
            elif g.y < 0:
                return Point(0, 1)
            continue
        return Point(0, 0)

    def shape_offset_frontiers_based(self, fake_shape=None, offset=Point(0, 0)):
        """Performs recursively offsets based on frontier rotation analysis.
        fake_shape is given as it is non-rotated yet. If it is detected collision
        before rotation happens, False is returned in order to avoid future
        rotations
        """
        fake_gameboard = self.gameboard_copy(mode="no_current_shape")
        if fake_shape is None:
            fake_shape = self.curr_shape.shape_copy()
            fake_shape.prepare_shape_for_rotation()
        updated_pivot = Point(
            fake_shape.gameboard_pivot_position.x + offset.x,
            fake_shape.gameboard_pivot_position.y + offset.y
        )
        for g, _ in self.fake_gameboard_points_and_shape_points(updated_pivot, fake_shape):
            try:
                if isinstance(fake_gameboard[g.x, g.y], Block):
                    return None  # can't rotate
            except IndexError:
                break
        fake_shape.gameboard_pivot_position = updated_pivot
        if not self.analyze_rotation_at_frontiers(updated_pivot, fake_shape):
            offset = self.frontier_offset(updated_pivot, fake_shape)
            updated_pivot = self.shape_offset_frontiers_based(fake_shape, offset)
        return updated_pivot  # can rotate

    def shape_offset_hood_based(self, fake_shape=None, offset=Point(0, 0)):
        """Temporally, returns True if there is not collision after rotation, i.e.,
        this method has the same function than analyze_rotation_at_neighborhood
        """
        if not fake_shape:
            fake_shape = self.curr_shape.shape_copy()
            fake_shape.prepare_shape_for_rotation()
        insertion_point = fake_shape.gameboard_pivot_position
        if self.analyze_rotation_at_neighborhood(insertion_point, fake_shape):
            return insertion_point
        return None

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
        fake_shape = self.curr_shape.shape_copy()
        fake_shape.prepare_shape_for_rotation()
        insertion_point = fake_shape.gameboard_pivot_position
        if not self.analyze_rotation_at_frontiers(insertion_point, fake_shape):
            candidate_point = self.shape_offset_frontiers_based()
        elif not self.analyze_rotation_at_neighborhood(insertion_point, fake_shape):
            candidate_point = self.shape_offset_hood_based()
        else:
            candidate_point = insertion_point
        if isinstance(candidate_point, Point):
            insertion_point = candidate_point
            self.clear_curr_shape()
            self.curr_shape.prepare_shape_for_rotation()
            self.put_shape_in_gameboard(insertion_point)
            return True
        return False

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
            if keyboard.is_pressed('r'):
                self.rotate_curr_shape_clockwise()
            yield from asyncio.sleep(2 * self.sleep_time)

    @asyncio.coroutine
    def trigger_timer(self):
        self.move_curr_shape_down()  # fall shape one time (natural falling)
        yield from asyncio.sleep(self.levels[self.level])

    @asyncio.coroutine
    def inline_rendering(self):
        while True:
            print(self, "SCORE = {}".format(self.game_score))
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
            self.after_curr_shape_cannot_fall()
            self._get_back_to_shape_generation = True

    def __trigger_loop(self):
        ioloop = asyncio.new_event_loop()
        ioloop.run_until_complete(self.__loop())
        ioloop.close()

    def game_over(self):
        self.is_game_over = True

    def remove_col(self, group):
        self.gameboard = np.delete(self.gameboard, list(group), axis=1)

    @property
    def current_increase(self):
        return 10 * self.level

    def score(self, group_lenght):
        self.game_score += self.current_increase * group_lenght

    def get_groups(self):
        groups, visited = [], {}
        for col in range(self.height):
            if isinstance(np.all(self.gameboard[:, col]), Block):
                if col - 1 not in visited:
                    visited[col] = len(groups)
                    groups.append({col})
                else:
                    visited[col] = len(groups) - 1
                    groups[visited[col - 1]].add(col)
        return groups

    def gameboard_padding(self):
        add = self.height - np.shape(self.gameboard)[1]
        self.gameboard = np.insert(
            self.gameboard,
            np.shape(self.gameboard)[1],
            np.zeros((add, np.shape(self.gameboard)[0]), dtype=int),
            axis=1,
        )

    def after_curr_shape_cannot_fall(self):
        if self.curr_shape.is_shape_on_gameboard_top(self.height):
            self.game_over()
            return
        groups = self.get_groups()
        for group in groups:
            self.remove_col(group)
            self.score(len(group))
        self.gameboard_padding()

    def _start(self):
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
        print('FINALE GAME OVER!')


class Render(Game):
    def __init__(self):
        super().__init__()


def main():
    Game()._start()
