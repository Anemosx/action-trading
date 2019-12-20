import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import random
from collections import namedtuple
from torch.distributions import Categorical



class SimpleDQN(torch.nn.Module):
    def __init__(self, channels, height, width, outputs):
        super(SimpleDQN, self).__init__()
        self.pre_head_dim = 16 # 32
        self.fc_net = nn.Sequential(
            nn.Linear(channels*height*width, 32), # 64
            nn.ELU(),
            nn.Linear(32, self.pre_head_dim), # 64
            nn.ELU()
        )

        self.action_head = nn.Linear(self.pre_head_dim, outputs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_net(x)
        return self.action_head(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.rng = random
        self.rng.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> []:
        return self.rng.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DqnAgent:
    def __init__(self,
                 observation_shape: [int],
                 number_of_actions: int,
                 gamma: float = 0.99,
                 epsilon_decay: float = 0.0001,
                 epsilon_min: float = 0.001,
                 mini_batch_size: int = 32,
                 warm_up_duration: int = 2000,
                 buffer_capacity: int = 50000,
                 target_update_period: int = 1000,
                 seed: int = 42) -> None:
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = 1
        self.number_of_actions = number_of_actions
        self.mini_batch_size = mini_batch_size
        self.warm_up_duration = warm_up_duration
        self.target_update_period = target_update_period
        self.memory = ReplayMemory(buffer_capacity, seed)
        self.policy_net = SimpleDQN(observation_shape[2],
                                    observation_shape[0],
                                    observation_shape[1],
                                    number_of_actions).to(self.device)
        self.target_net = SimpleDQN(observation_shape[2],
                                    observation_shape[0],
                                    observation_shape[1],
                                    number_of_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0005)
        self.training_count = 0

    def save(self, state, action, next_state, reward, done):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        action = torch.tensor([action], device=self.device, dtype=torch.long)
        if done:
            next_state = None
        else:
            next_state = torch.tensor([next_state], device=self.device, dtype=torch.float32)
        reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        self.memory.push(state, action, next_state, reward)

    def train(self):
        if len(self.memory) < self.warm_up_duration:
            return
        transitions = self.memory.sample(self.mini_batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        next_state_values = torch.zeros(self.mini_batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + next_state_values * self.gamma

        self.optimizer.zero_grad()
        loss = f.smooth_l1_loss(state_action_values, expected_state_action_values)  # huber loss
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        self.training_count += 1
        if self.training_count % self.target_update_period is 0:
            print("loss before target network update: {:.5f}".format(loss))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

    def policy(self, state, agent_index=None):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.number_of_actions, (1,)).item()
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            return self.policy_net(state).max(1)[1].item()

    def compute_q_values(self, state):
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            return self.policy_net(state).numpy()

    def save_weights(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_weights(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
        self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

def make_dqn_agent(params, observation_shape, nb_actions):
    agent = DqnAgent(
        observation_shape=observation_shape,
        number_of_actions=nb_actions,
        gamma=params.gamma,
        epsilon_decay=params.epsilon_decay,
        epsilon_min=params.epsilon_min,
        mini_batch_size=params.mini_batch_size,
        warm_up_duration=params.warm_up_duration,
        buffer_capacity=params.buffer_capacity,
        target_update_period=params.target_update_period,
        seed=1337)
    return agent