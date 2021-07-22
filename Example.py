import GameEngine as TicTacToe
import Neural_Network as Neural_Network
import torch

# Runs on GPU if possible
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

env = TicTacToe.GameEngine()

# Initalize the neural nets and load presaved weights
test_1_hidden_size = Neural_Network.Network(env, 16, 504)
test_2_hidden_size = Neural_Network.Network(env, 16, 504)
test_3_hidden_size = Neural_Network.Network(env, 50)
test_1_hidden_size.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/opp_nn_tic_tac_toe_50_v4"))
test_2_hidden_size.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_tic_tac_toe_50_v4"))
test_3_hidden_size.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_tic_tac_toe_target_50_expanded"))
test_1_hidden_size.to(device)
test_2_hidden_size.to(device)
test_3_hidden_size.to(device)

obs = env.reset()

# Sets each score to 0
test_1_hidden_size_wins_going_first = 0
test_2_hidden_size_wins_going_second = 0
test_3_hidden_size_wins_going_second = 0
ties_1_going_first = 0
test_1_hidden_size_wins_going_second = 0
test_2_hidden_size_wins_going_first = 0
test_3_hidden_size_wins_going_first = 0
ties_2_going_first = 0
ties_3_going_first = 0
ties_3_going_first_round_2 = 0
test_3_hidden_size_wins_going_first_round_2 = 0
test_3_hidden_size_wins_going_second_round_2 = 0
test_2_hidden_size_wins_going_second_round_2 = 0
test_2_hidden_size_wins_going_first_round_2 = 0
ties_2_going_first_round_2 = 0


# Compares Test 1 and Test 2 with test 1 going first across all 9 starting squares
for x in range(1, 10):
    obs = env.reset()
    env.update_square("X", x)
    while not env.is_winner("X") and not env.is_winner("O"):
        env.game_board.print_grid()
        print("")

        obs = env.get_ai_state('O')
        env.update_square('O', test_2_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")
        if env.is_winner("O"):
            test_2_hidden_size_wins_going_second = test_2_hidden_size_wins_going_second + 1
            break

        obs = env.get_ai_state('X')
        env.update_square('X', test_1_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")

        if env.is_winner("X"):
            test_1_hidden_size_wins_going_first = test_1_hidden_size_wins_going_first + 1
            break

        if len(env.list_of_valid_moves()) == 0:
            ties_1_going_first = ties_1_going_first + 1
            break

print("")
print("")

# Compares Test 1 and Test 2 with test 2 going first across all 9 starting squares
for x in range(1, 10):
    obs = env.reset()
    env.update_square("X", x)
    while not env.is_winner("X") and not env.is_winner("O"):
        env.game_board.print_grid()
        print("")

        obs = env.get_ai_state('O')
        env.update_square('O', test_1_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")
        if env.is_winner("O"):
            test_1_hidden_size_wins_going_second = test_1_hidden_size_wins_going_second + 1
            break

        obs = env.get_ai_state('X')
        env.update_square('X', test_2_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")

        if env.is_winner("X"):
            test_2_hidden_size_wins_going_first = test_2_hidden_size_wins_going_first + 1
            break

        if len(env.list_of_valid_moves()) == 0:
            ties_2_going_first = ties_2_going_first + 1
            break


ties_1_going_first_round_2 = 0
test_1_hidden_size_wins_going_first_round_2 = 0
test_1_hidden_size_wins_going_second_round_2 = 0

# Compares Test 1 and Test 3 with test 1 going first across all 9 starting squares
for x in range(1, 10):
    obs = env.reset()
    env.update_square("X", x)
    while not env.is_winner("X") and not env.is_winner("O"):
        env.game_board.print_grid()
        print("")

        obs = env.get_ai_state('O')
        env.update_square('O', test_3_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")
        if env.is_winner("O"):
            test_3_hidden_size_wins_going_second = test_3_hidden_size_wins_going_second + 1
            break

        obs = env.get_ai_state('X')
        env.update_square('X', test_1_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")

        if env.is_winner("X"):
            test_1_hidden_size_wins_going_first_round_2 = test_1_hidden_size_wins_going_first_round_2 + 1
            break

        if len(env.list_of_valid_moves()) == 0:
            ties_1_going_first_round_2 = ties_1_going_first_round_2 + 1
            break

print("")
print("")

# Compares Test 1 and Test 3 with test 3 going first across all 9 starting squares
for x in range(1, 10):
    obs = env.reset()
    env.update_square("X", x)
    while not env.is_winner("X") and not env.is_winner("O"):
        env.game_board.print_grid()
        print("")

        obs = env.get_ai_state('O')
        env.update_square('O', test_1_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")
        if env.is_winner("O"):
            test_1_hidden_size_wins_going_second_round_2 = test_1_hidden_size_wins_going_second_round_2 + 1
            break

        obs = env.get_ai_state('X')
        env.update_square('X', test_3_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")

        if env.is_winner("X"):
            test_3_hidden_size_wins_going_first = test_3_hidden_size_wins_going_first + 1
            break

        if len(env.list_of_valid_moves()) == 0:
            ties_3_going_first = ties_3_going_first + 1
            break

# Compares Test 2 and Test 3 with test 2 going first across all 9 starting squares
for x in range(1, 10):
    obs = env.reset()
    env.update_square("X", x)
    while not env.is_winner("X") and not env.is_winner("O"):
        env.game_board.print_grid()
        print("")

        obs = env.get_ai_state('O')
        env.update_square('O', test_3_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")
        if env.is_winner("O"):
            test_3_hidden_size_wins_going_second_round_2 = test_3_hidden_size_wins_going_second_round_2 + 1
            break

        obs = env.get_ai_state('X')
        env.update_square('X', test_2_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")

        if env.is_winner("X"):
            test_2_hidden_size_wins_going_first_round_2 = test_2_hidden_size_wins_going_first_round_2 + 1
            break

        if len(env.list_of_valid_moves()) == 0:
            ties_2_going_first_round_2 = ties_2_going_first_round_2 + 1
            break

print("")
print("")

# Compares Test 2 and Test 3 with test 3 going first across all 9 starting squares
for x in range(1, 10):
    obs = env.reset()
    env.update_square("X", x)
    while not env.is_winner("X") and not env.is_winner("O"):
        env.game_board.print_grid()
        print("")

        obs = env.get_ai_state('O')
        env.update_square('O', test_2_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")
        if env.is_winner("O"):
            test_2_hidden_size_wins_going_second_round_2 = test_2_hidden_size_wins_going_second_round_2 + 1
            break

        obs = env.get_ai_state('X')
        env.update_square('X', test_3_hidden_size.act(obs, env, device) + 1)
        env.game_board.print_grid()
        print("")

        if env.is_winner("X"):
            test_3_hidden_size_wins_going_first_round_2 = test_3_hidden_size_wins_going_first_round_2 + 1
            break

        if len(env.list_of_valid_moves()) == 0:
            ties_3_going_first_round_2 = ties_3_going_first_round_2 + 1
            break


print("1 Wins Going First vs 2: ", test_1_hidden_size_wins_going_first)
print("2 Wins Going Second: ", test_2_hidden_size_wins_going_second)
print("Ties 1 Going First vs 2: ", ties_1_going_first)

print("")

print("1 Wins Going Second vs 2: ", test_1_hidden_size_wins_going_second)
print("@ Wins Going First: ", test_2_hidden_size_wins_going_first)
print("Ties 2 Going First: ", ties_2_going_first)

print("")
print("")

print("1 Wins Going First vs 3: ", test_1_hidden_size_wins_going_first_round_2)
print("3 Wins Going Second: ", test_3_hidden_size_wins_going_second)
print("Ties 3 Going First: ", ties_1_going_first_round_2)

print("")

print("1 Wins Going Second: vs 3: ", test_1_hidden_size_wins_going_second_round_2)
print("3 Wins Going First: ", test_3_hidden_size_wins_going_first)
print("Ties 3 Going First: ", ties_3_going_first)

print("")
print("")

print("2 Wins Going First vs 3: ", test_2_hidden_size_wins_going_first_round_2)
print("3 Wins Going Second Round 2: ", test_3_hidden_size_wins_going_second_round_2)
print("Ties 2 Going First Round 2: ", ties_2_going_first_round_2)

print("")

print("2 Wins Going Second: vs 3: ", test_2_hidden_size_wins_going_second_round_2)
print("3 Wins Going First: Round 2 ", test_3_hidden_size_wins_going_first_round_2)
print("Ties 3 Going First Round 2: ", ties_3_going_first_round_2)