import GameBoard as Board
import numpy as np
import  random as rand

class GameEngine:
    def __init__(self):
        self.game_board = Board.TicTacToe()

    def display(self):
        self.game_board.print_grid()

    def update_square(self, player_symbol, grid_coordinate):
            if self.game_board.check_valid_input(player_symbol, grid_coordinate):
                self.game_board.set_grid_square(player_symbol, grid_coordinate)
                return True
            else:
                return False

    def ai_selection(self, selection_weights):
        resized_selection_weights = np.reshape(selection_weights, (-1, 3))
        for y in range(3):
            for x in range(3):
                if not self.game_board.check_valid_input("X", x, y):
                    resized_selection_weights[x, y] = 0

        highest_weight = np.where(resized_selection_weights == np.amax(resized_selection_weights))
        print('Tuple of arrays returned : ', highest_weight)
        print('List of coordinates of maximum value in Numpy array : ')
        # zip the 2 arrays to get the exact coordinates
        selected_move = list(zip(highest_weight[0], highest_weight[1]))
        return highest_weight[0], highest_weight[1]

    def win_indexes(self, n):
        # Rows
        for r in range(n):
            yield [(r, c) for c in range(n)]
        # Columns
        for c in range(n):
            yield [(r, c) for r in range(n)]
        # Diagonal top left to bottom right
        yield [(i, i) for i in range(n)]
        # Diagonal top right to bottom left
        yield [(i, n - 1 - i) for i in range(n)]

    def is_winner(self, decorator):
        n = len(self.game_board.get_grid())
        for indexes in self.win_indexes(n):
            if all(self.game_board.get_grid()[r][c] == decorator for r, c in indexes):
                return True
        return False

    def ai_random_move(self, player):
        self.update_square(player, rand.choice(self.game_board.get_valid_squares()))