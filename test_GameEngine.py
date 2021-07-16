from unittest import TestCase
from GameEngine import GameEngine
import pytest
import numpy as np


@pytest.fixture
def test_game_engine():
    return GameEngine()


def test_update_square_invalid_selections(test_game_engine):
    assert test_game_engine.update_square("x", -1) is False, "Tic Tac Toe did  allow a square to be written too under the minimum square 1"
    assert test_game_engine.update_square("x", 'a') is False, "Tic Tac Toe did  allow a square to be written to a non number square"
    assert test_game_engine.update_square("x", 0) is False, "Tic Tac Toe did  allow a square to be written too over the maximum square 9"


def test_update_square_normal_value(test_game_engine):
    assert test_game_engine.update_square("X", 5) is True, "Tic Tac Toe did not allow a X to be written"
    assert test_game_engine.update_square("O", 2) is True, "Tic Tac Toe did not allow a O to be written"


def test_update_square_overwrite(test_game_engine):
    assert test_game_engine.update_square("O", 5) is True,"Tic Tac Toe did not allow a blank space to be written"
    assert test_game_engine.update_square("O", 5) is False, "Tic Tac Toe allowed a space to be overwritten"


def test_is_winner_first_row(test_game_engine):
    test_game_engine.update_square("X", 1)
    test_game_engine.update_square("X", 2)
    test_game_engine.update_square("X", 3)
    result = test_game_engine.is_winner('X')
    assert  test_game_engine.is_winner("X") is True, "Top row not detected as winning"


def test_is_winner_first_column(test_game_engine):
    test_game_engine.update_square("X", 1)
    test_game_engine.update_square("X", 4)
    test_game_engine.update_square("X", 7)
    result = test_game_engine.is_winner('X')
    assert  test_game_engine.is_winner("X") is True, "First column not detected as winning"


def test_is_winner_diagonal_bottom_to_top(test_game_engine):
    test_game_engine.update_square("X", 7)
    test_game_engine.update_square("X", 5)
    test_game_engine.update_square("X", 3)
    assert  test_game_engine.is_winner("X") is True, "Diagonal bottom to top not detected as winning"


def test_is_winner_diagonal_top_to_bottom(test_game_engine):
    test_game_engine.update_square("X", 1)
    test_game_engine.update_square("X", 5)
    test_game_engine.update_square("X", 9)
    assert  test_game_engine.is_winner("X") is True, "Diagonal top to bottom not detected as winning"


def test_no_winner_empty_board(test_game_engine):
    assert  test_game_engine.is_winner("X") is False, "Empty board detecting as winning"


def test_no_winner(test_game_engine):
    test_game_engine.update_square("X", 1)
    test_game_engine.update_square("O", 5)
    test_game_engine.update_square("X", 9)
    assert  test_game_engine.is_winner("X") is False, "Detecting winning in a mixed line"

def test_full_board(test_game_engine):
    test_game_engine.update_square("X", 1)
    test_game_engine.update_square("O", 2)
    test_game_engine.update_square("O", 3)
    test_game_engine.update_square("O", 4)
    test_game_engine.update_square("O", 5)
    test_game_engine.update_square("O", 6)
    test_game_engine.update_square("O", 7)
    test_game_engine.update_square("O", 8)
    test_game_engine.update_square("X", 9)
    assert len(test_game_engine.list_of_valid_moves()) == 0, "Full board has valid moves"
