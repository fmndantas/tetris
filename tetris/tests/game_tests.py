import unittest

import tetris
from tetris import *

Point = tetris.Point


def get_ip(game: tetris.Game) -> Point:
    return game.initial_points[game.crr_sh.shape_id]


class GameTests(unittest.TestCase):
    def setUp(self):
        self.game = tetris.Game(width=10, height=10)
        self.shapes = self.game.shape_generator.shapes

    def test_shape_generation(self):
        self.assertEqual(self.game.crr_sh, None)
        self.game.generate_next_shape()
        self.assertFalse(self.game.crr_sh is self.shapes[self.game.crr_sh.shape_id - 1])
        self.game.crr_sh.grid[0, 0] = 'change test'
        self.assertNotEqual(self.shapes[self.game.crr_sh.shape_id - 1].grid[0, 0], 'change test')

    def test_put_shape_in_gameboard(self):
        for i in range(7):
            self.game.crr_sh = self.shapes[i].shape_copy()
            ip = get_ip(self.game)
            self.game.put_shape_in_gameboard(ip)
            left = self.game.crr_sh.left
            right = self.game.crr_sh.right
            top = self.game.crr_sh.top
            bottom = self.game.crr_sh.bottom
            gb_slice = [
                self.game.gameboard[i, j] for i in range(left, right + 1) for j in range(bottom, top + 1)
            ]
            rows = self.game.crr_sh.width
            cols = self.game.crr_sh.height
            sh_slice = [
                self.game.crr_sh.grid[i, j] for i in range(rows) for j in range(cols)
            ]
            self.assertEqual(gb_slice, sh_slice)
            self.game.gameboard = np.zeros_like(self.game.gameboard)

    def test_prepare_shape_for_rotation(self):
        shape = self.shapes[0].shape_copy()
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

        not_pivot_points = [i.gameboard_position for i in shape.grid.flatten() if not i.is_pivot]
        self.assertTrue(np.all(not_pivot_points) is None)

    def test_deep_shape_copy(self):
        source = self.shapes[0].shape_copy()
        destiny = source.shape_copy()
        for key in destiny.__dict__.keys():
            if key != 'grid':
                self.assertEqual(destiny.__getattribute__(key), source.__getattribute__(key))
            # else:
            #     self.assertEqual(destiny.grid[0], source.grid[0])
        destiny.grid[0, 0] = 'change test'
        self.assertNotEqual(source.grid[0, 0], 'change test')
        self.assertEqual(destiny.grid[0, 0], 'change test')
        self.assertFalse(source.grid[1, 0] is destiny.grid[1, 0])

    def test_analyze_rotation_at_frontiers(self):
        self.game.crr_sh = self.shapes[0].shape_copy()
        self.game.put_shape_in_gameboard(self.game.initial_shape_point_for_gameboard)
        fake_shape = self.game.crr_sh.shape_copy()
        fake_shape.prepare_shape_for_rotation()
        self.assertFalse(self.game.analyze_rotation_at_frontiers(fake_shape.gameboard_pivot_position, fake_shape))

    def test_analyze_rotation_at_hood(self):
        self.game.crr_sh = self.shapes[0].shape_copy()
        self.game.put_shape_in_gameboard(Point(4, 7))  # considering neighborhood
        fake_shape = self.game.crr_sh.shape_copy()
        fake_shape.prepare_shape_for_rotation()
        self.assertTrue(self.game.analyze_rotation_at_neighborhood(fake_shape.gameboard_pivot_position, fake_shape))
        self.game.clear_crr_sh()
        self.game.crr_sh = self.shapes[3]
        self.game.put_shape_in_gameboard(Point(4, 6))
        self.game.crr_sh = self.shapes[0]
        self.game.put_shape_in_gameboard(Point(4, 7))
        fake_shape = self.game.crr_sh.shape_copy()
        fake_shape.prepare_shape_for_rotation()
        self.assertFalse(self.game.analyze_rotation_at_neighborhood(fake_shape.gameboard_pivot_position, fake_shape))

    def test_shape_offset_frontiers_based(self):
        self.game.crr_sh = self.shapes[0].shape_copy()
        self.game.put_shape_in_gameboard(self.game.initial_shape_point_for_gameboard)  # top case
        self.assertEqual(Point(4, 7), self.game.shape_offset_frontiers_based())
        self.game.clear_crr_sh()

        self.game.crr_sh = self.shapes[1]  # left case
        self.game.put_shape_in_gameboard(Point(0, 5))
        self.assertEqual(Point(1, 5), self.game.shape_offset_frontiers_based())
        self.game.clear_crr_sh()

        self.game.crr_sh = self.shapes[2]  # right case
        self.game.put_shape_in_gameboard(Point(9, 5))
        self.assertEqual(Point(8, 5), self.game.shape_offset_frontiers_based())
        self.game.clear_crr_sh()

        self.game.crr_sh = self.shapes[0]  # bottom case
        self.game.put_shape_in_gameboard(Point(5, 0))
        self.assertEqual(Point(5, 1), self.game.shape_offset_frontiers_based())
        self.game.clear_crr_sh()

    def test_rect_rotation(self):
        self.game.crr_sh = self.shapes[0].shape_copy()

        # Case: frontier at top -> translate pivot two squares down
        self.game.put_shape_in_gameboard(self.game.initial_shape_point_for_gameboard)
        self.game.rotate_curr_shape_clockwise()
        self.assertEqual(Point(4, 7), self.game.crr_sh.gameboard_pivot_position)
        self.game.reset()

        # Case: blocking square shape -> shape won't rotate
        self.game.crr_sh = self.shapes[3].shape_copy()
        self.game.put_shape_in_gameboard(Point(5, 5))
        self.game.crr_sh = self.shapes[0].shape_copy()
        self.game.put_shape_in_gameboard(Point(5, 6))
        status = self.game.rotate_curr_shape_clockwise()
        self.assertFalse(status)
        self.assertEqual(self.game.crr_sh.gameboard_pivot_position, Point(5, 6))
        self.game.reset()

        # Case: frontier at right and blocking shape -> shape won't rotate
        self.game.crr_sh = self.shapes[3].shape_copy()
        self.game.put_shape_in_gameboard(Point(8, 1))
        self.game.crr_sh = self.shapes[0].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(9, 1))
        status = self.game.rotate_curr_shape_clockwise()
        self.assertFalse(status)

    def test_frontier_offset(self):
        self.game.crr_sh = self.shapes[0].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(9, 4))
        self.assertEqual(
            Point(-1, 0),
            self.game.frontier_offset(Point(9, 4), self.game.crr_sh),
        )  # right case
        self.game.clear_crr_sh()
        self.game.put_shape_in_gameboard(Point(0, 4))
        self.assertEqual(
            Point(1, 0),
            self.game.frontier_offset(Point(0, 4), self.game.crr_sh)
        )  # left case

    def test_curr_shape_offset_without_block_collision(self):
        self.game.crr_sh = self.shapes[0].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()

        self.game.put_shape_in_gameboard(Point(9, 4))  # to left case
        self.assertEqual(Point(7, 4), self.game.shape_offset_frontiers_based())

        self.game.clear_crr_sh()

        self.game.put_shape_in_gameboard(Point(0, 4))  # to right case
        self.assertEqual(Point(1, 4), self.game.shape_offset_frontiers_based())

        self.game.clear_crr_sh()

        self.game.crr_sh = self.shapes[0].shape_copy()  # to down case
        self.game.put_shape_in_gameboard(Point(4, 9))
        self.assertEqual(Point(4, 7), self.game.shape_offset_frontiers_based())

        self.game.clear_crr_sh()

        self.game.put_shape_in_gameboard(Point(5, 5))
        self.assertEqual(Point(5, 5), self.game.shape_offset_frontiers_based())  # no need translation case

    def test_curr_shape_offset_with_block_collision(self):
        ell = self.shapes[4].shape_copy()
        rect = self.shapes[0].shape_copy()

        self.game.crr_sh = ell
        self.game.put_shape_in_gameboard(Point(7, 4))
        self.game.crr_sh = rect
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(9, 4))
        self.assertEqual(None, self.game.shape_offset_frontiers_based())

        self.game.gameboard = np.zeros_like(self.game.gameboard)
        self.game.crr_sh = ell
        self.game.put_shape_in_gameboard(Point(5, 4))
        self.game.crr_sh = rect
        self.game.put_shape_in_gameboard(Point(9, 4))
        self.assertEqual(Point(7, 4), self.game.shape_offset_frontiers_based())

    def test_rect_right_translation_at_insert_point(self):
        self.game.crr_sh = self.shapes[0].shape_copy()
        ip = get_ip(self.game)
        self.game.put_shape_in_gameboard(ip)
        while self.game.crr_sh.right < self.game.width - 1:
            self.game.move_curr_shape_right()
        self.game.move_curr_shape_right()
        self.assertEqual(self.game.crr_sh.right, 9)

    def test_rect_left_translation_at_insert_point(self):
        self.game.crr_sh = self.shapes[0].shape_copy()
        ip = get_ip(self.game)
        self.game.put_shape_in_gameboard(ip)
        while self.game.crr_sh.left > 0:
            self.game.move_curr_shape_left()
        self.game.move_curr_shape_left()
        self.assertEqual(self.game.crr_sh.left, 0)

    def test_get_group(self):
        self.game.gameboard.T[:3] = Block(color='w')
        self.game.gameboard.T[8:10] = Block(color='w')
        self.assertEqual([{0, 1, 2}, {8, 9}], self.game.get_groups())

    def test_remove_col_and_score(self):
        self.game.gameboard.T[:3] = Block(color='w')
        self.game.crr_sh = self.shapes[0]
        self.game.put_shape_in_gameboard(Point(5, 3))
        groups = self.game.get_groups()
        for group in groups:
            self.game.remove_col(group)
            self.game.score(len(group))
        self.game.gameboard_padding()
        self.assertEqual(80, self.game.game_score)

    def test_ell_shape(self):
        self.game.crr_sh = self.shapes[3]
        self.game.put_shape_in_gameboard(Point(5, 2))
        self.game.put_shape_in_gameboard(Point(8, 2))
        self.game.crr_sh = self.shapes[1].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(6, 3))
        self.game.move_curr_shape_down()
        self.game.move_curr_shape_down()

    def test_lower_block_per_col(self):
        self.game.crr_sh = self.shapes[1].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(5, 5))
        gen = self.game.crr_sh.lower_block_per_col
        self.assertEqual(Point(4, 6), next(gen))
        self.assertEqual(Point(5, 4), next(gen))

    def test_most_left_block_per_row(self):
        self.game.crr_sh = self.shapes[1].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(1, 1))
        gen = self.game.crr_sh.extremed_block_per_row(side='l')
        self.assertEqual(Point(1, 0), next(gen))
        self.assertEqual(Point(1, 1), next(gen))
        self.assertEqual(Point(0, 2), next(gen))

    def test_most_right_block_per_row(self):
        self.game.crr_sh = self.shapes[1].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(1, 1))
        gen = self.game.crr_sh.extremed_block_per_row(side='r')
        self.assertEqual(Point(1, 0), next(gen))
        self.assertEqual(Point(1, 1), next(gen))
        self.assertEqual(Point(1, 2), next(gen))

    def test_can_move_right(self):
        self.game.crr_sh = self.shapes[1].shape_copy()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.crr_sh.prepare_shape_for_rotation()
        self.game.put_shape_in_gameboard(Point(1, 1))
        self.assertTrue(self.game.crr_sh.can_shape_move_right(self.game.gameboard))


if __name__ == '__main__':
    unittest.main()
