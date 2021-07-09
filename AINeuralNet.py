from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import GameEngine as TicTacToe

GAMMA = 0.99
BATCH_SIZE = 1
BUFFER_SIZE = 500000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 0.0
EPSILON_END = 0.0
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 500



class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        #in_features = int(np.prod(env.observation_space.shape))
        in_features = 27

        self.net = nn.Sequential(
            nn.Linear(in_features, 150),
            nn.Tanh(),
            nn.Linear(150, env.action_space.n)
        )

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        possible_values = env.list_of_valid_moves()

        for x in range(1, 10):
            if x not in possible_values:
                q_values[0][x - 1] = -10.0

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action

#env = gym.make(GameEngine)
env = TicTacToe.GameEngine()

replay_buffer = deque(maxlen=BUFFER_SIZE)

rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)

target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-3)

#initalize replay buffer
obs = env.reset()
for __ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, rew, done, info = env.step(action, "X")
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    obs = new_obs
    if done:
        obs = env.reset()

# Main training loop
obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)
        env.update_square('X', action+1)
        if step > 250000:
            env.game_board.print_grid()

    new_obs, rew, done, info = env.step(action, 'X')
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)

    obs = new_obs

    episode_reward += rew
    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0


    #Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute Targets
    target_q_values = target_net(new_obses_t)
    possible_values = env.list_of_valid_moves()

    '''for x in range(1, 10):
        if x not in possible_values:
            target_q_values[0][x-1] = -10.0
    '''

    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

    #Compute Loss
    q_values = online_net(obses_t)

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Actual Gradiant Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    #logging
    if step %1000 == 0:
        print()
        print('Step ', step)
        print('Avg Rew', np.mean(rew_buffer))
        print(env.game_board.print_grid())

'''
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(27, 243)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(243, 9)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x
        
'''