import GameBoard as board

game_board = board.TicTacToe()


if game_board.check_valid_input('X', 0, 1):
    game_board.set_grid_square('X', 0, 1)
else:
    print("Invalid")
game_board.print_grid()

if game_board.check_valid_input('X', 0, 0):
    game_board.set_grid_square('X', 0, 0)
else:
    print("Invalid")

game_board.reset_board()

if game_board.check_valid_input('X', 0, 0):
    game_board.set_grid_square('X', 0, 0)
else:
    print("Invalid")
game_board.print_grid()

if game_board.check_valid_input('X', 0, 1):
    game_board.set_grid_square('X', 0, 1)
else:
    print("Invalid")

print(game_board.is_winner("X"))

game_board.set_grid_square('X', 1, 1)
game_board.set_grid_square('X', 2, 2)

game_board.print_grid()
print(game_board.is_winner("X"))

print("Get Test")