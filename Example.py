import GameEngine as TicTacToe
import Neural_Network as Neural_Network
import torch


env = TicTacToe.GameEngine()

test1 = Neural_Network.Network(env)
test2 = Neural_Network.Network(env)
test2.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_initial_tic_tac_toe"))
test1.load_state_dict(torch.load("/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_tic_tac_toe_target"))

obs = env.reset()
print(test1.act(obs, env))
env.update_square('X', test1.act(obs, env) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('X')
env.update_square('X', test2.act(obs, env) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('X')
env.update_square('X', test2.act(obs, env) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('X')
env.update_square('X', test2.act(obs, env) + 1)
env.game_board.print_grid()
print("")

obs = env.get_ai_state('O')
env.update_square('O', test2.act(obs, env) + 1)
env.game_board.print_grid()