import os
import numpy as np
from copy import deepcopy
from envs.smartfactory import Smartfactory
from common_utils.utils import export_video
from agent import build_agent
from dotmap import DotMap
import json
import common_utils.drawing_util as drawing_util
import pandas as pd

class Trade:

    def __init__(self, agent_1, agent_2, mark_up=1.0):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.agents = [agent_1, agent_2]
        self.mark_up = mark_up

    # trade suggestion

    def trade_suggestion_n_steps(self, env, observations, actions, combined_frames=None, info_values=None):

        trading, greedy = env.check_trading(actions)

        if not trading:
            observations, r, done, info = env.step(actions)
            return observations, r, done, info, trading

        else:

            rewards = np.zeros(2)
            done = False
            info = None

            # get agents Q-values

            q_vals_a1 = self.agents[0].compute_q_values(observations[0])
            q_vals_a2 = self.agents[1].compute_q_values(observations[1])
            q_vals = [q_vals_a1, q_vals_a2]

            # if Q-value is higher than other

                # calculate compensation reward depending on Q-value

                # enable trade action

    # make trade action:

        # change color of agent to indicate trade possibility

        # set flag to remember the trade related actions and reward

    # follow suggestion:

        # depending on Q-value make decision to follow suggestion

        # remember steps of the suggested actions

        # enable following suggested action steps

    # pay reward to agent:

        # check if actions have been followed

        # make decision on paying agent depending on Q-Value

        # exchange reward


# test trading

def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    c = 0
    t = 0
    policy_random = False
    episodes = 1
    episode_steps = 100

    ep_columns = ['episode', 'contracting', 'reward', 'number_contracts', 'episode_steps']
    for i_ag in range(params.nb_agents):
        ag_columns = ['reward_a{}'.format(i_ag),
                      'accumulated_transfer_a{}'.format(i_ag)]
        ep_columns += ag_columns
    df = pd.DataFrame(columns=ep_columns)

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       trading=t,
                       contracting=c,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks
                       )

    processor = env.SmartfactoryProcessor()