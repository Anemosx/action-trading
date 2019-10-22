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

    def __init__(self, agent_1, agent_2, n_trade_steps=1, mark_up=1.0):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.agents = [agent_1, agent_2]
        self.n_trade_steps = n_trade_steps
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

            # calculate compensation reward depending on Q-value

            for i_agent, agent in enumerate(self.agents):
                if not greedy[i_agent]:
                    transfer = np.maximum(np.max(q_vals[i_agent]), 0) * self.mark_up

                    #env.agents[(i_agent + 1) % 2].episode_debts += transfer

            # enable trade-action by adding to existing actions

            for t_step in range(self.n_trade_steps):
                for i_agent, agent in enumerate(self.agents):
                    if greedy[i_agent]:
                        actions.append(self.agents[i_agent].forward(observations[i_agent]))
                    else:
                        actions.append(np.argmin(q_vals[i_agent]))

                observations, r, done, info = env.step(actions)
                observations = deepcopy(observations)

                r, transfer = self.clarify_reward(env=env, rewards=r, trade_reward=transfer)
                rewards += r

                if combined_frames is not None:
                    if info_values is not None:
                        for i, agent in enumerate(env.agents):
                            info_values[i]['a{}-reward'.format(i)] = r[i]
                            info_values[i]['a{}-episode_debts'.format(i)] = env.agents[i].episode_debts
                            info_values[i]['trading'] = 1
                            info_values[i]['a{}-greedy'.format(i)] = greedy[i]
                            info_values[i]['a{}-q_max'.format(i)] = np.max(q_vals[i])

                    combined_frames = drawing_util.render_combined_frames(combined_frames, env, info_values)

                if any([agent.done for agent in env.agents]):
                    break

            return observations, rewards, done, info, trading


    # make trade action:



        # set flag to remember the trade related actions and reward

    # follow suggestion:

        # depending on Q-value make decision to follow suggestion

        # remember steps of the suggested actions

        # enable following suggested action steps

    # pay reward to agent depending on Q-Value:

    def clarify_reward(self, env, rewards, trade_reward):

        transfer = [0, 0]

        # check if actions have been followed



        # make decision on paying agent depending on Q-Value

        # exchange reward

        return r, transfer


# test trading

def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    t = 0
    policy_random = False
    episodes = 1
    episode_steps = 100

    ep_columns = ['episode', 'trading', 'reward', 'number_trades', 'episode_steps']
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
                       contracting=0,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks
                       )

    processor = env.SmartfactoryProcessor()

    trade = None

    if t != 0:
        trading_agents = []
        for i in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=params.one_step_trading_action, processor=processor)
            agent.load_weights('experiments/20191015-09-39-50/run-0/contracting-0/dqn_weights-agent-{}.h5f'.format(i))
            trading_agents.append(agent)
        trade = Trade(agent_1=trading_agents[0], agent_2=trading_agents[1], n_trade_steps=params.trading_steps, mark_up=params.mark_up)

        agents = []
        for i_agent in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=env.nb_contracting_actions, processor=processor)
            agents.append(agent)
            agents[i_agent].load_weights(
                'experiments/20191017-15-11-23/step-penalty-0.001/run-0.001/contracting-{}/dqn_weights-agent-{}.h5f'.format(
                    0, i_agent))

        combined_frames = []
        for i_episode in range(episodes):
            observations = env.reset()
            episode_rewards = np.zeros(params.nb_agents)
            accumulated_transfer = np.zeros(params.nb_agents)
            trading = False

            if trade is not None:
                q_vals_a1 = trade.agents[0].compute_q_values(observations[0])
                q_vals_a2 = trade.agents[1].compute_q_values(observations[1])
            else:
                q_vals_a1 = agents[0].compute_q_values(observations[0])
                q_vals_a2 = agents[1].compute_q_values(observations[1])
            q_vals = [q_vals_a1, q_vals_a2]

            info_values = [{'a{}-reward'.format(i): 0.0,
                            'a{}-episode_debts'.format(i): 0.0,
                            'trading': 0,
                            'a{}-greedy'.format(i): 0,
                            'a{}-q_max'.format(i): np.max(q_vals[i]),
                            'a{}-done'.format(i): env.agents[i].done
                            } for i in range(params.nb_agents)]

            combined_frames = drawing_util.render_combined_frames(combined_frames, env, info_values, observations)

            for i_step in range(episode_steps):
                actions = []
                for i_ag in range(params.nb_agents):
                    if not env.agents[i_ag].done:
                        if not policy_random:
                            actions.append(agents[i_ag].forward(observations[i_ag]))
                        else:
                            actions.append(np.random.randint(0, env.nb_actions))
                    else:
                        actions.append(0)
                if trade is not None:
                    observations, r, done, info, trading = trade.trade_suggestion_n_steps(env, observations, actions, combined_frames, info_values)
                else:
                    observations, r, done, info = env.step(actions)

                observations = deepcopy(observations)

                # payout depending on Q-Value (if Q-Value of payout is highest)
                # has to be modified since agent has the choice

                if trade is not None and not any([agent.done for agent in env.agents]):
                    r, transfer = trade.clarify_reward(env=env, rewards=r)
                    accumulated_transfer += transfer
                episode_rewards += r

                if not trading:
                    if trade is not None:
                        q_vals_a1 = trade.agents[0].compute_q_values(observations[0])
                        q_vals_a2 = trade.agents[1].compute_q_values(observations[1])
                    else:
                        q_vals_a1 = agents[0].compute_q_values(observations[0])
                        q_vals_a2 = agents[1].compute_q_values(observations[1])
                    q_vals = [q_vals_a1, q_vals_a2]
                    for i, agent in enumerate(env.agents):
                        info_values[i]['a{}-reward'.format(i)] = r[i]
                        info_values[i]['trading'] = trading
                        info_values[i]['a{}-greedy'.format(i)] = 0
                        info_values[i]['a{}-q_max'.format(i)] = np.max(q_vals[i])
                        info_values[i]['a{}-done'.format(i)] = env.agents[i].done

                    combined_frames = drawing_util.render_combined_frames(combined_frames, env, info_values,
                                                                          observations)

                if done:
                    ep_stats = [i_episode, (trade is not None), np.sum(episode_rewards), 0,
                                episode_steps]
                    for i_ag in range(len(agents)):
                        ag_stats = [episode_rewards[i_ag], accumulated_transfer[i_ag]]
                        ep_stats += ag_stats
                    df.loc[i_episode] = ep_stats
                    break

            df.to_csv(os.path.join('test-values-contracting-c-{}.csv'.format(0)))
            export_video('Smart-Factory-Trading.mp4', combined_frames, None)