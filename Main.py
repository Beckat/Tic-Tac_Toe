import GameEngine as TicTacToe
import numpy as np

game = TicTacToe.GameEngine()
player = "X"

player_has_won = False
player_has_updated_board = False

player_count = input("How many players (1-2) ")
if not player_count == "1" and not player_count == "2":
    player_count = "1"

while not player_has_won:
    game.display()

    while player_has_updated_board == False:
        if player == "X" or player_count == "2":
            selection = input("What grid would you like to place your marked in (1-9)? ")

            # attempts to update the board with player selection, updates the square if possible asks again if not
            player_has_updated_board = game.update_square(player, selection)
            if player_has_updated_board == False:
                print("\nThat is an invalid selection\n")
        else:
            game.ai_random_move("O")
            print("\n")
            player_has_updated_board = True

    player_has_won = game.is_winner(player)
    player_has_updated_board = False

    if player_has_won == False:
        if player == "X":
            player = "O"
        else:
            player = "X"
    else:
        print("Player " + player + " has won")
        game.display()
