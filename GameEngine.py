import GameBoard as Board
import numpy as np
import  random as rand
import gym


class GameEngine(gym.Env):
    """
    Holds logic to run the tic-tac-toe game using the GameBoard class
    """
    def __init__(self):
        self.game_board = Board.TicTacToe()
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Discrete(27)

    def display(self):
        """
        prints the board to the terminal
        """
        self.game_board.print_grid()

    def update_square(self, player_symbol, grid_coordinate):
            """
            If the square is valid update with the symbol provided in squares 1-9
            1 is the first square in the upper left and 9 is the last square in the lower right
            :param player_symbol: a single character string
            :param grid_coordinate: integer 1 to 9
            :return: returns boolean if successfully updated
            """
            if self.game_board.check_valid_input(player_symbol, grid_coordinate):
                self.game_board.set_grid_square(player_symbol, grid_coordinate)
                return True
            else:
                return False

    def ai_selection(self, selection_weights):
        """
        Place holder
        :param selection_weights:
        :return:
        """
        for x in range(selection_weights.size):
            if not self.game_board.check_valid_input("X", x + 1):
                selection_weights[x] = 0
        return str(np.argmax(selection_weights, axis=0) + 1)

    def win_indexes(self, n):
        """
        Calculates the ways a player can win three in a row, three in a column or three diagonally
        :param n: integer the size of the board
        """
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
        """
        Checks to see if the selected character has won
        :param decorator: Symbol of the player checking if they won
        :return: Boolean true if the character has a winning board
        """
        n = len(self.game_board.get_grid())
        for indexes in self.win_indexes(n):
            if all(self.game_board.get_grid()[r][c] == decorator for r, c in indexes):
                return True
        return False

    def ai_random_move(self, player):
        """

        :param player:
        """
        self.update_square(player, rand.choice(self.game_board.get_valid_squares()))

    def get_ai_state(self, decorator='X'):
        """
        Gets the input values to the neural net from the game board
        the first 9 values are for blank spaces, the second set of 9 are for the AIs pieces and the
        final 9 are for the opponents pieces
        :param decorator: What character represents the AI's pieces
        :return: returns the 27 long numpy array to feed into the neural net
        """
        ai_observation_state = np.full((27), 0)
        counter = 0
        for row in self.game_board.get_grid():
            for square in row:
                if square == '.':
                    ai_observation_state[counter] = 1
                elif square == 'X':
                    if decorator == 'X':
                        ai_observation_state[counter + 9] = 1
                    else:
                        ai_observation_state[counter + 18] = 1
                elif decorator == 'O':
                    ai_observation_state[counter + 9] = 1
                else:
                    ai_observation_state[counter + 18] = 1
                counter = counter + 1
        return ai_observation_state

    def list_of_valid_moves(self):
        """
        Returns which squares can be updated with a new player move
        :return: Array of valid move ids 1-9
        """
        valid_moves = []
        for x in range(1,10):
            if self.game_board.check_valid_input('X', x):
                valid_moves.append(x)
        return valid_moves

    def step(self, action, decorator,ai_move=-1):
        """
        Updates the rewards values, game state, if the game is complete and if the AI won, lost, or tied
        :param action: Placeholder
        :param decorator: What piece the AI is playing
        :return: current state of the board to feed into the neural net, the reward the net recieved,
        if the game state is done with a winner, if the AI won,lost or tied
        """
        done = False
        finish_state = ""
        if decorator == "X":
            ai_decorator = "O"
        else:
            ai_decorator = "X"

        reward = 0

        if self.is_winner(decorator):
            reward = 1
            done = True
            finish_state = "Win"
        elif self.list_of_valid_moves() == []:
            done = True
            if reward == 0:
                reward = .25
                finish_state = "Tie"
        else:
            random_num = rand.randint(0, 3)
            if ai_move == -1:
                if self.game_board.check_valid_input(ai_decorator, 5) and random_num == 0:
                    self.update_square(ai_decorator, 5)
                else:
                    self.ai_random_move(ai_decorator)
            elif random_num > 0:
                self.update_square(ai_decorator, ai_move)
            else:
                self.ai_random_move(ai_decorator)

            if self.is_winner(ai_decorator):
                reward = -1
                done = True
                finish_state = "Lose"

        info = {finish_state}
        state = self.get_ai_state(decorator)

        return state, reward, done, info

    def reset(self):
        """
        Returns the gameboard back to blank and returns that new state to the neural net
        :return: the newly updated blank board state
        """
        self.game_board.reset_board()
        state = self.get_ai_state()

        return state