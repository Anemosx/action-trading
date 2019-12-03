import gym
import numpy as np
from itertools import count
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class A2C(nn.Module):
    def __init__(self):
        super(A2C, self).__init__()
        self.shared_net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU()
        )

        self.action_head = nn.Linear(128, 2)  # actor
        self.value_head = nn.Linear(128, 1)  # critic
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.shared_net(x)

        action_probs = F.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_probs, state_values

    def select_action(self, state, train=True):
        if not train:
            self.eval()  # to avoid dropout/batch norm during testing
            with torch.no_grad():
                action, log_prob, state_value = self.__select_action(state)
            self.train()  # always be explicit
        else:
            action, log_prob, state_value = self.__select_action(state)
            self.saved_actions.append(SavedAction(log_prob, state_value))
        return action.item()

    def __select_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob, state_value = self.forward(state)

        m = Categorical(action_prob)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action, log_prob, state_value

    def save(self):
        pass


class Trainer:
    def __init__(self,
                 env,
                 actor_critic,
                 gamma,
                 num_epochs,
                 max_steps_episode=1000,
                 render=False):
        self.env = env
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.num_epochs = num_epochs
        self.max_steps_episode = max_steps_episode
        self.render = render

    def train_loop(self):
        for epoch in range(self.num_epochs):

            s = self.env.reset()
            r_episode = 0

            for step_num in range(self.max_steps_episode):
                a = self.actor_critic.select_action(s)

                s, r, done, _ = self.env.step(a)

                if self.render:
                    self.env.render()

                self.actor_critic.rewards.append(r)
                r_episode += r

                if done:
                    break

            self.end_episode()

    def end_episode(self):
        pass
