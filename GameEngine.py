import GameBoard as Board
import numpy as np
import  random as rand
import gym


class GameEngine(gym.Env):
    def __init__(self):
        self.game_board = Board.TicTacToe()
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Discrete(27)

    def display(self):
        self.game_board.print_grid()

    def update_square(self, player_symbol, grid_coordinate):
            if self.game_board.check_valid_input(player_symbol, grid_coordinate):
                self.game_board.set_grid_square(player_symbol, grid_coordinate)
                return True
            else:
                return False

    def ai_selection(self, selection_weights):
        for x in range(selection_weights.size):
            if not self.game_board.check_valid_input("X", x + 1):
                selection_weights[x] = 0
        return str(np.argmax(selection_weights, axis=0) + 1)

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
        n = len(self.game_board.get_grid())
        for indexes in self.win_indexes(n):
            if all(self.game_board.get_grid()[r][c] == decorator for r, c in indexes):
                return True
        return False

    def ai_random_move(self, player):
        self.update_square(player, rand.choice(self.game_board.get_valid_squares()))

    def get_ai_state(self):
        ai_observation_state = np.full((27), 0)
        counter = 0
        for row in self.game_board.get_grid():
            for square in row:
                if square == '.':
                    ai_observation_state[counter] = 1
                elif square == 'X':
                    ai_observation_state[counter + 7] = 1
                else:
                    ai_observation_state[counter + 14] = 1
                counter = counter + 1
        return ai_observation_state

    def list_of_valid_moves(self):
        valid_moves = []
        for x in range(1,10):
            if self.game_board.check_valid_input('X', x):
                valid_moves.append(x)
        return valid_moves

    def step(self, action, decorator):
        done = False

        reward = .01

        if self.is_winner(decorator):
            reward = 1
            done = True
        elif self.list_of_valid_moves() == []:
            done = True
            if reward == .01:
                reward = .25
        elif decorator == 'X':
            self.ai_random_move('O')
            if self.is_winner('O'):
                reward = -1
                done = True
        else:
            self.ai_random_move('X')
            if self.is_winner('X'):
                reward = -1
                done = True

        info = {}
        state = self.get_ai_state()

        return state, reward, done, info

    def reset(self):
        self.game_board.reset_board()
        state = self.get_ai_state()

        return state