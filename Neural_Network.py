from torch import nn
import torch


class Network(nn.Module):
    """
    Builds out a one or two hidden layer network
    Has a size 27 input row to represent 0 for each row in not blank "." and 1 for each blank square
    The same for X and O squares
    """
    def __init__(self, env, hidden_size=243, hidden_size_2=0):
        super().__init__()

        #in_features = int(np.prod(env.observation_space.shape))
        in_features = 27

        if hidden_size_2 == 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, env.action_space.n)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, hidden_size_2),
                nn.Tanh(),
                nn.Linear(hidden_size_2, env.action_space.n)
            )

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.net(x)

    def act(self, obs, env, device=torch.device("cpu")):
        """
        Finds the action with the highest expected Q value
        If the action would be invalid set the Q value to -1000
        :param obs:
        :param env:
        :param device:
        :return:
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        q_values = self(obs_t.unsqueeze(0))
        possible_values = env.list_of_valid_moves()

        for x in range(1, 10):
            if x not in possible_values:
                q_values[0][x - 1] = -1000.0

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action
