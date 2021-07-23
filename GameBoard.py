import numpy as np


class TicTacToe:
    """
    Initializes the game board
    """
    def __init__(self, num_players = 1):
        self.tic_tac_toe_grid = [ ['.']*3 for i in range(3)]
        self.tic_tac_toe_grid = np.asarray(self.tic_tac_toe_grid)

    def reset_board(self):
        """
        Sets all squares back to default "blanks" '.'
        """
        self.tic_tac_toe_grid = [ ['.']*3 for i in range(3)]
        self.tic_tac_toe_grid = np.asarray(self.tic_tac_toe_grid)

    def print_grid(self):
        """
        Prints each square as a tic-tac-toe grid
        """
        for y in range(3):
            for x in range(3):
                print(self.tic_tac_toe_grid[x][y], end='')
                if x == 2:
                    print("")

    def get_grid(self):
        return self.tic_tac_toe_grid

    def set_grid_square(self, value, grid_coordinate):
        """
        Updates the square 1-9 with the X or O
        :param value:
        :param grid_coordinate:
        """
        self.tic_tac_toe_grid[self.get_x_coordinate(grid_coordinate)][self.get_y_coordinate(grid_coordinate)] = value

    def check_valid_input(self, value, grid_coordinate):
        """
        Checks if the square selected 1-9 is an empty square and in the valid range
        :param value:
        :param grid_coordinate:
        :return:
        """
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
        """
        Takes a square selection 1-9 and finds what the x row is matching that
        1 is row 0, 4 would be row 1
        :param grid_coordinate:
        :return:
        """
        return int((int(grid_coordinate) - 1) % 3)

    def get_y_coordinate(self, grid_coordinate):
        """
        Takes a square selection 1-9 and finds what the y column is matching that
        1 is column 0, 3 would be column 2
        :param grid_coordinate:
        :return:
        """
        return int((int(grid_coordinate) - 1) / 3)

    def get_valid_squares(self):
        """
        Returns the empty values [1-9] squares in a list
        :return:
        """
        valid_squares = []
        for square in range(self.tic_tac_toe_grid.size):
            if self.check_valid_input("X", square + 1):
                valid_squares.append(square + 1)
        return valid_squares

    def get_1d_array_of_board(self):

        return self.tic_tac_toe_grid.flatten()