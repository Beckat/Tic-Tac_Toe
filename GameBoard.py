class TicTacToe:

    def __init__(self, num_players = 0):
        self.tic_tac_toe_grid = [ ['.']*3 for i in range(3)]
        self.turn = "x_turn"
        self.num_players_tic_tac_toe = num_players

    def reset_board(self):
        self.tic_tac_toe_grid = [['.'] * 3 for i in range(3)]
        self.turn = "x_turn"

    def print_grid(self):
        for y in range(3):
            for x in range(3):
                print(self.tic_tac_toe_grid[x][y], end='')
                if x == 2:
                    print("")

    def get_grid(self):
        return self.tic_tac_toe_grid

    def set_grid_square(self, value, x_coordinate, y_coordinate):
        self.tic_tac_toe_grid[x_coordinate][y_coordinate] = value


    def check_valid_input(self, value, x_coordinate, y_coordinate):
        if value == "X" and self.tic_tac_toe_grid[x_coordinate][y_coordinate] == "." or value == "O" and self.tic_tac_toe_grid[x_coordinate][y_coordinate] == ".":
            return True
        else:
            return False

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
        n = len(self.tic_tac_toe_grid)
        for indexes in self.win_indexes(n):
            if all(self.tic_tac_toe_grid[r][c] == decorator for r, c in indexes):
                return True
        return False





