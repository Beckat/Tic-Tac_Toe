import GameEngine as TicTacToe
import Neural_Network as Neural_Network
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

env = TicTacToe.GameEngine()

test1 = Neural_Network.Network(env)
test2 = Neural_Network.Network(env, 150)
test1.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_tic_tac_toe_target_30"))
test2.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_tic_tac_toe"))
test1.to(device)
test2.to(device)

obs = env.reset()
print(test1.act(obs, env, device))
env.update_square('X', test1.act(obs, env, device) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env, device) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('X')
env.update_square('X', test2.act(obs, env, device) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env, device) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('X')
env.update_square('X', test2.act(obs, env, device) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env, device) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('X')
env.update_square('X', test2.act(obs, env, device) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env, device) + 1)
env.game_board.print_grid()