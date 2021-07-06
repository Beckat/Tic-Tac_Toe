import GameBoard as board
import numpy as np


class GameEngine:
    def __init__(self):
        game_board = board

    def display(self):
        game_board.print_grid()

    def update_square(self, player_symbol, x_coordinate, y_coordinate):
        if game_board.check_valid_input(player_symbol,x_coordinate,y_coordinate):
            game_board.set_grid_square(player_symbol,x_coordinate,y_coordinate)
            return "Success"
        else:
            return "Failure"

    def ai_selection(self, selection_weights):
        resized_selection_weights = np.reshape(selection_weights, (-1, 3))
        for y in range(3):
            for x in range(3):
                if not game_board.check_valid_input("X", x, y):
                    resized_selection_weights[x, y] = 0

        highest_weight = np.where(resized_selection_weights == np.amax(resized_selection_weights))
        print('Tuple of arrays returned : ', highest_weight)
        print('List of coordinates of maximum value in Numpy array : ')
        # zip the 2 arrays to get the exact coordinates
        selected_move = list(zip(highest_weight[0], highest_weight[1]))
        return highest_weight[0], highest_weight[1]

