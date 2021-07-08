from unittest import TestCase
from GameEngine import GameEngine
import numpy as np


class TestGameEngine(TestCase):
    def setUp(self) -> None:
        self.test_game_engine = GameEngine()

    def test_update_square_negative_number(self):
        expected = False
        result = self.test_game_engine.update_square("X", -1)
        self.assertEqual(result, expected, "Updated with a negative number")

    def test_update_square_not_a_number(self):
        expected = False
        result = self.test_game_engine.update_square("X", 'a')
        self.assertEqual(result, expected, "Updated with not a number")

    def test_update_square_past_maximum_number(self):
        expected = False
        result = self.test_game_engine.update_square("X", 10)
        self.assertEqual(result, expected, "Updated with ano out of bounds maximum number")

    def test_update_square_normal_value_X(self):
        expected = True
        result = self.test_game_engine.update_square("X", 5)
        self.assertEqual(result, expected, "Failed to update with an X a valid number")

    def test_update_square_normal_value_O(self):
        expected = True
        result = self.test_game_engine.update_square("O", 5)
        self.assertEqual(result, expected, "Failed to update with an O a valid number")

    def test_update_square_overwrite_existing_value(self):
        expected = False
        result = self.test_game_engine.update_square("X", 5)
        result = self.test_game_engine.update_square("X", 5)
        self.assertEqual(result, expected, "Overwrote an existing space")

    def test_is_winner_top_row(self):
        expected = True
        self.test_game_engine.update_square("X", 1)
        self.test_game_engine.update_square("X", 2)
        self.test_game_engine.update_square("X", 3)
        result = self.test_game_engine.is_winner('X')
        self.assertEqual(result, expected, "Top row not detected as winning")

    def test_is_winner_first_column(self):
        expected = True
        self.test_game_engine.update_square("X", 1)
        self.test_game_engine.update_square("X", 4)
        self.test_game_engine.update_square("X", 7)
        result = self.test_game_engine.is_winner('X')
        self.assertEqual(result, expected, "First column not detected as winning")

    def test_is_winner_diagnal_bottom_to_top(self):
        expected = True
        self.test_game_engine.update_square("X", 7)
        self.test_game_engine.update_square("X", 5)
        self.test_game_engine.update_square("X", 3)
        result = self.test_game_engine.is_winner('X')
        self.assertEqual(result, expected, "First column not detected as winning")

    def test_is_winner_diagonal_top_to_bottom(self):
        expected = True
        self.test_game_engine.update_square("X", 1)
        self.test_game_engine.update_square("X", 5)
        self.test_game_engine.update_square("X", 9)
        result = self.test_game_engine.is_winner('X')
        self.assertEqual(result, expected, "First column not detected as winning")

    def test_is_winner_O(self):
        expected = True
        self.test_game_engine.update_square("O", 1)
        self.test_game_engine.update_square("O", 4)
        self.test_game_engine.update_square("O", 7)
        result = self.test_game_engine.is_winner('O')
        self.assertEqual(result, expected, "O not detected as winning")

    def test_no_winner_empty_board(self):
        expect = False
        result = self.test_game_engine.is_winner("X")
        self.assertEqual(result, expect, "Winning on empty board")

    def test_no_winner(self):
        expect = False
        self.test_game_engine.update_square("O", 1)
        self.test_game_engine.update_square("X", 4)
        self.test_game_engine.update_square("O", 7)
        result = self.test_game_engine.is_winner("X")
        self.assertEqual(result, expect, "Winning with X and O forming line")

'''
    def test_initial_board(self):
        base_board =np.full((3, 3), ['.'],dtype=str)
        #base_board = np.empty(shape=(3,3), dtype='<U4')
        #for item in base_board:
        #    item = '.'
        base_board.fill('.')
        np.testing.assert_equal(self.test_game_engine., base_board, "Incorrect Board Intialization")
'''
#class initialize(TestGameEngine):



''' 
    def test_ai_selection(self):
        self.fail()

    def test_ai_random_move(self):
        self.fail()
'''