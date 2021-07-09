import numpy as np
import gym

class TicTacToe:

    def __init__(self, num_players = 1):
        self.tic_tac_toe_grid = [ ['.']*3 for i in range(3)]
        self.tic_tac_toe_grid = np.asarray(self.tic_tac_toe_grid)
        self.turn = "x_turn"
        self.num_players_tic_tac_toe = num_players


    def reset_board(self):
        self.tic_tac_toe_grid = [ ['.']*3 for i in range(3)]
        self.tic_tac_toe_grid = np.asarray(self.tic_tac_toe_grid)

    def print_grid(self):
        for y in range(3):
            for x in range(3):
                print(self.tic_tac_toe_grid[x][y], end='')
                if x == 2:
                    print("")

    def get_grid(self):
        return self.tic_tac_toe_grid

    def set_grid_square(self, value, grid_coordinate):
        self.tic_tac_toe_grid[self.get_x_coordinate(grid_coordinate)][self.get_y_coordinate(grid_coordinate)] = value

    def check_valid_input(self, value, grid_coordinate):
        try:
            if int(grid_coordinate) in range(1, 10):
                if value == "X" and self.tic_tac_toe_grid[self.get_x_coordinate(grid_coordinate)][self.get_y_coordinate(grid_coordinate)] == "." or value == "O" and self.tic_tac_toe_grid[self.get_x_coordinate(grid_coordinate)][self.get_y_coordinate(grid_coordinate)] == ".":
                    return True
                else:
                    return False
            else:
                return False
        except Exception as ex:
            return False

    def get_x_coordinate(self, grid_coordinate):
        return int((int(grid_coordinate) - 1) % 3)

    def get_y_coordinate(self, grid_coordinate):
        return int((int(grid_coordinate) - 1) / 3)

    def get_valid_squares(self):
        valid_squares = []
        for square in range(self.tic_tac_toe_grid.size):
            if self.check_valid_input("X", square + 1):
                valid_squares.append(square + 1)
        return valid_squares

    def get_1d_array_of_board(self):
        return self.tic_tac_toe_grid.flatten()