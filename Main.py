import GameEngine as TicTacToe
import numpy as np
import Neural_Network as Neural_Network
import torch

game = TicTacToe.GameEngine()
player = "."

game_is_over = False
player_has_updated_board = False
player_count = 0
player_turn = True

ai_X = Neural_Network.Network(game, 9, 135)
ai_O = Neural_Network.Network(game, 9, 135)
ai_X.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_tic_tac_toe_50_v3"))
ai_O.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/opp_nn_tic_tac_toe_50_v3"))

while not player_count == "1" and not player_count == "2":
    player_count = input("How many players (1-2) ")

if player_count == "1":
    while not player == "X" and not player == "O":
        player = input("Which piece would you like to play (X-O) ")
else:
    player = "X"

if player == "O":
    player_turn = False

''' Make Player Able to Pick Selection'''

while not game_is_over:
    game.display()

    while player_has_updated_board == False:
        if player_turn == True or player_count == "2":
            selection = input("What grid would you like to place your marked in (1-9)? ")

            # attempts to update the board with player selection, updates the square if possible asks again if not
            player_has_updated_board = game.update_square(player, selection)
            if player_has_updated_board == False:
                print("\nThat is an invalid selection\n")
            if player_count == "2":
                if player == "X":
                    player = "O"
                else:
                    player = "X"
            else:
                player_turn = False
        else:
            if player == "X":
                obs = game.get_ai_state("O")
                game.update_square('O', ai_O.act(obs, game) + 1)
            else:
                obs = game.get_ai_state("X")
                game.update_square("X", ai_X.act(obs, game) + 1)
            print("\n")
            player_has_updated_board = True
            player_turn = True

    game_is_over = game.is_winner(player)
    player_has_updated_board = False

    if game_is_over:
        print("Player " + player + " has won")
        game.display()
    elif len(game.list_of_valid_moves()) == 0:
        print("The game is a tie")
        game.display()
        game_is_over = True
