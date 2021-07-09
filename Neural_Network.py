from torch import nn
import torch


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

    def act(self, obs, env):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))
        possible_values = env.list_of_valid_moves()

        for x in range(1, 10):
            if x not in possible_values:
                q_values[0][x - 1] = -10.0

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action
