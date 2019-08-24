import unittest

import numpy as np

import tetris

Point = tetris.Point


def get_ip(game: tetris.Game) -> Point:
    return game.initial_points[game.curr_shape.shape_id]


class GameTests(unittest.TestCase):
    def setUp(self):
        self.game = tetris.Game()
        self.shapes = self.game.shape_generator.shapes

    def test_shape_generation(self):
        self.assertEqual(self.game.curr_shape, None)
        self.game.generate_next_shape()
        self.assertNotEqual(self.game.curr_shape, None)

    def test_put_shape_in_gameboard(self):
        for i in range(7):
            self.game.curr_shape = self.shapes[i]
            ip = get_ip(self.game)
            self.game.put_shape_in_gameboard(ip)
            left = self.game.curr_shape.left
            right = self.game.curr_shape.right
            top = self.game.curr_shape.top
            bottom = self.game.curr_shape.bottom
            gb_slice = [
                self.game.gameboard[i, j] for i in range(left, right + 1) for j in range(bottom, top + 1)
            ]
            rows = self.game.curr_shape.width
            cols = self.game.curr_shape.height
            sh_slice = [
                self.game.curr_shape.grid[i, j] for i in range(rows) for j in range(cols)
            ]
            self.assertEqual(gb_slice, sh_slice)
            self.game.gameboard = np.zeros_like(self.game.gameboard)

    def test_deep_shape_copy(self):
        source = self.shapes[0]
        destiny = source.shape_copy()
        for key in destiny.__dict__.keys():
            if key != 'grid':
                self.assertEqual(destiny.__getattribute__(key), source.__getattribute__(key))
            else:
                self.assertTrue(np.all(source.grid.flatten() == destiny.grid.flatten()))

    def test_prepare_shape_for_rotation(self):
        # (2, 0) -> (0, 1)
        # (4, 5) -> (4, 6)
        shape = self.shapes[0]
        shape.grid[0, 0].gameboard_position = Point(2, 5)
        shape.grid[1, 0].gameboard_position = Point(3, 5)
        shape.grid[2, 0].gameboard_position = Point(4, 5)
        shape.grid[2, 0].is_pivot = True
        shape.grid[3, 0].gameboard_position = Point(5, 5)

        self.assertEqual(shape.pivot, Point(2, 0))
        self.assertEqual(Point(4, 5), shape.gameboard_pivot_position)

        shape.prepare_shape_for_rotation()
        self.assertEqual(Point(0, 1), shape.pivot)

        self.assertEqual(Point(4, 5), shape.gameboard_pivot_position)

    def test_rect_rotation_at_insert_point(self):
        self.game.curr_shape = self.shapes[0]
        ip = get_ip(self.game)

        self.game.curr_shape.grid[0, 0].gameboard_position = Point(2, 9)
        self.game.curr_shape.grid[1, 0].gameboard_position = Point(3, 9)
        self.game.curr_shape.grid[2, 0].gameboard_position = Point(4, 9)
        self.game.curr_shape.grid[2, 0].is_pivot = True
        self.game.curr_shape.grid[3, 0].gameboard_position = Point(5, 9)

        self.assertEqual(ip, self.game.curr_shape.gameboard_pivot_position)
        self.assertFalse(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position),
                         'Shape should not be rotable here')
        self.game.put_shape_in_gameboard(ip)
        self.game.rotate_curr_shape_clockwise()  # There is a conditional verification in rotate... So, test will pass

    def test_rect_horizontal_rotation_checking_at_various_places(self):
        self.game.curr_shape = self.shapes[1]  # ell
        ip = Point(0, 5)
        self.game.put_shape_in_gameboard(ip)
        self.assertFalse(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position))
        self.game.move_curr_shape_right()
        self.assertTrue(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position))
        self.game.rotate_curr_shape_clockwise()

    def test_rect_vertical_rotation_checking_at_various_places(self):
        self.game.curr_shape = self.shapes[0]
        ip = get_ip(self.game)
        self.game.put_shape_in_gameboard(ip)

        self.assertFalse(
            self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position)
        )

        self.game.gameboard = np.zeros_like(self.game.gameboard)  # now, one down
        self.game.put_shape_in_gameboard(ip)
        self.game.move_curr_shape_down()
        self.assertFalse(
            self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position)
        )

        self.game.gameboard = np.zeros_like(self.game.gameboard)  # now, two down
        self.game.put_shape_in_gameboard(ip)
        self.game.move_curr_shape_down()
        self.game.move_curr_shape_down()
        self.assertTrue(
            self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position)
        )

    def test_rect_rotation_at_various_places(self):
        self.game.curr_shape = self.shapes[0]
        self.game.put_shape_in_gameboard(Point(4, 7))
        self.assertEqual(self.game.curr_shape.gameboard_pivot_position, Point(4, 7))
        self.assertTrue(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position))
        self.game.rotate_curr_shape_clockwise()

    def test_rect_rotation_with_block_collision(self):
        self.game.curr_shape = self.shapes[3]  # square
        self.game.put_shape_in_gameboard(Point(6, 1))
        self.game.curr_shape = self.shapes[3]  # square
        self.game.put_shape_in_gameboard(Point(3, 1))
        self.game.curr_shape = self.shapes[0]  # rect
        self.game.put_shape_in_gameboard(Point(4, 2))
        self.assertTrue(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position))
        self.game.rotate_curr_shape_clockwise()
        print(self.game)
        self.assertFalse(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position))

    def test_ell_rotation_with_block_collision(self):
        self.game.curr_shape = self.shapes[3]  # square
        self.game.put_shape_in_gameboard(Point(2, 1))
        self.game.curr_shape = self.shapes[3]  # square
        self.game.put_shape_in_gameboard(Point(7, 1))
        self.game.curr_shape = self.shapes[2]  # ell
        self.game.put_shape_in_gameboard(Point(4, 0))
        self.assertTrue(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position))
        self.game.rotate_curr_shape_clockwise()
        print(self.game)
        self.assertFalse(self.game.is_curr_shape_rotable(self.game.curr_shape.gameboard_pivot_position))

    def test_rect_right_translation_at_insert_point(self):
        self.game.curr_shape = self.shapes[0]
        ip = get_ip(self.game)
        self.game.put_shape_in_gameboard(ip)
        while self.game.curr_shape.right < self.game.width - 1:
            self.game.move_curr_shape_right()
        self.game.move_curr_shape_right()
        self.assertEqual(self.game.curr_shape.right, 9)

    def test_rect_left_translation_at_insert_point(self):
        self.game.curr_shape = self.shapes[0]
        ip = get_ip(self.game)
        self.game.put_shape_in_gameboard(ip)
        while self.game.curr_shape.left > 0:
            self.game.move_curr_shape_left()
        self.game.move_curr_shape_left()
        self.assertEqual(self.game.curr_shape.left, 0)


if __name__ == '__main__':
    unittest.main()
