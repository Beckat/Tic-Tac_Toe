from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import GameEngine as TicTacToe
import Neural_Network

GAMMA = 0.99
BATCH_SIZE = 25
BUFFER_SIZE = 500000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 0.0
EPSILON_END = 0.0
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 500
wins = 0
losses = 0
ties = 0

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


#env = gym.make(GameEngine)
env = TicTacToe.GameEngine()

replay_buffer = deque(maxlen=BUFFER_SIZE)
opp_replay_buffer = deque(maxlen=BUFFER_SIZE)

rew_buffer = deque([0.0], maxlen=100)
opp_rew_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0
opp_episode_reward = 0.0

online_net = Neural_Network.Network(env, 144, 112)

target_net = Neural_Network.Network(env, 144, 112)
online_net.to(device)
target_net.to(device)

opp_online_net = Neural_Network.Network(env, 144, 112)
opp_target_net = Neural_Network.Network(env, 144, 112)
opp_online_net.to(device)
opp_target_net.to(device)

target_net.load_state_dict(online_net.state_dict())


optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-6,)
opp_optimizer = torch.optim.Adam(opp_online_net.parameters(), lr=5e-6,)

print(online_net)

#initalize replay buffer
obs = env.reset()
opp_obs = env.reset()
for __ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    new_obs, rew, done, info = env.step(action, "X")
    opp_rew = rew[1]
    rew = rew[0]
    opp_new_obs = new_obs[1]
    new_obs = new_obs[0]
    transition = (obs, action, rew, done, new_obs)
    opp_transition = (opp_obs, action, opp_rew, done, opp_new_obs)
    replay_buffer.append(transition)
    opp_replay_buffer.append(opp_transition)

    obs = new_obs
    opp_obs = opp_new_obs
    if done:
        obs = env.reset()

# Main training loop
obs = env.reset()
opp_obs = env.reset()

for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
    rnd_sample = random.random()

    if rnd_sample <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs, env, device)
        env.update_square('X', action+1)
        opposition_obs = env.get_ai_state('O')
        opp_action = opp_online_net.act(opposition_obs, env, device) + 1

    new_obs, rew, done, info = env.step(action, 'X', opp_action)
    opp_rew = rew[1]
    rew = rew[0]
    opp_new_obs = new_obs[1]
    new_obs = new_obs[0]

    if next(iter(info)) == "Win":
        wins = wins + 1
    if next(iter(info)) == "Lose":
        losses = losses + 1
    if next(iter(info)) == "Tie":
        ties = ties + 1
    transition = (obs, action, rew, done, new_obs)
    opp_transition = (opp_obs, opp_action, opp_rew, done, opp_new_obs)
    replay_buffer.append(transition)
    opp_replay_buffer.append(opp_transition)

    opp_obs = opp_new_obs
    obs = new_obs

    episode_reward += rew
    opp_episode_reward += opp_rew
    if done:
        obs = env.reset()
        opp_obs = env.reset()
        rew_buffer.append(episode_reward)
        opp_rew_buffer.append(opp_episode_reward)
        episode_reward = 0.0
        opp_episode_reward = 0.0


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
    target_q_values = target_net(new_obses_t.to(device))
    possible_values = env.list_of_valid_moves()

    '''for x in range(1, 10):
        if x not in possible_values:
            target_q_values[0][x-1] = -10.0
    '''

    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values.cpu()

    #Compute Loss
    q_values = online_net(obses_t.to(device))

    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t.to(device)).to(device)

    loss = nn.functional.smooth_l1_loss(action_q_values.to(device), targets.to(device)).to(device)

    # Actual Gradiant Descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())




    #Start Gradient Step
    opp_transitions = random.sample(opp_replay_buffer, BATCH_SIZE)

    opp_obses = np.asarray([t[0] for t in opp_transitions])
    opp_actions = np.asarray([t[1] for t in opp_transitions])
    opp_rews = np.asarray([t[2] for t in opp_transitions])
    opp_dones = np.asarray([t[3] for t in opp_transitions])
    opp_new_obses = np.asarray([t[4] for t in opp_transitions])

    opp_obses_t = torch.as_tensor(opp_obses, dtype=torch.float32)
    opp_actions_t = torch.as_tensor(opp_actions, dtype=torch.int64).unsqueeze(-1)
    opp_rews_t = torch.as_tensor(opp_rews, dtype=torch.float32).unsqueeze(-1)
    opp_dones_t = torch.as_tensor(opp_dones, dtype=torch.float32).unsqueeze(-1)
    opp_new_obses_t = torch.as_tensor(opp_new_obses, dtype=torch.float32)

    # Compute Targets
    opp_target_q_values = opp_target_net(opp_new_obses_t.to(device))
    opp_possible_values = env.list_of_valid_moves()

    opp_max_target_q_values = opp_target_q_values.max(dim=1, keepdim=True)[0]

    opp_targets = opp_rews_t + GAMMA * (1 - opp_dones_t) * opp_max_target_q_values.cpu()

    #Compute Loss
    opp_q_values = opp_online_net(opp_obses_t.to(device))

    opp_action_q_values = torch.gather(input=opp_q_values, dim=1, index=opp_actions_t.to(device)).to(device)

    opp_loss = nn.functional.smooth_l1_loss(opp_action_q_values.to(device), opp_targets.to(device)).to(device)

    # Actual Gradiant Descent
    opp_optimizer.zero_grad()
    opp_loss.backward()
    opp_optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())



    #logging
    if step %1000 == 0:
        print()
        print('Step ', step)
        print('Avg Rew', np.mean(rew_buffer))
        print("Wins ", wins)
        print("Loses ", losses)
        print("Ties ", ties)
        wins = 0
        losses = 0
        ties = 0
        print(env.game_board.print_grid())
        if step < 2000:
            torch.save(online_net.state_dict(), "/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_initial_tic_tac_toe_two_layers_v2")
            torch.save(opp_online_net.state_dict(), "/home/danthom1704/PycharmProjects/Tic-Tac_toe/opp_nn_initial_tic_tac_toe_two_layers_v2")
        if step > 10000:
            torch.save(online_net.state_dict(), "/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_tic_tac_toe_two_layers_v2")
            torch.save(target_net.state_dict(), "/home/danthom1704/PycharmProjects/Tic-Tac_toe/nn_target_tic_tac_toe_two_layers_v2")
            torch.save(opp_online_net.state_dict(), "/home/danthom1704/PycharmProjects/Tic-Tac_toe/opp_nn_tic_tac_toe_two_layers_v2")
            torch.save(opp_target_net.state_dict(), "/home/danthom1704/PycharmProjects/Tic-Tac_toe/opp_nn_target_tic_tac_toe_two_layers_v2")
