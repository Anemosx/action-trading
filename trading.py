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


def setup_action_space(step, trading_steps, tr_action_space):
    extended_space = []
    if tr_action_space is None:
        tr_action_space = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
    if step == 0:
        if trading_steps != 0:
            no_tr_action_space = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
            for i in range(len(no_tr_action_space)):
                no_tr_action_space[i] = no_tr_action_space[i] + [0.0, 0.0] * trading_steps
            tr_action_space = no_tr_action_space + tr_action_space
        return tr_action_space
    if step > 0:
        for i_actions in range(len(tr_action_space)):
            extended_space.append(tr_action_space[i_actions] + [0.0, 1.0])
            extended_space.append(tr_action_space[i_actions] + [0.0, -1.0])
            extended_space.append(tr_action_space[i_actions] + [-1.0, 0.0])
            extended_space.append(tr_action_space[i_actions] + [1.0, 0.0])
        extended_space = setup_action_space(step - 1, trading_steps, extended_space)
        return extended_space


class Trade:

    def __init__(self, agent_1, agent_2, n_trade_steps, mark_up):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.agents = [agent_1, agent_2]
        self.n_trade_steps = n_trade_steps
        self.mark_up = mark_up

    # trade suggestion

    def trade_n_steps(self, env, observations, actions):

        trading = env.check_trading(actions)
        transfer = 0

        if not trading:
            observations, r, done, info = env.step(actions)
            return observations, r, done, info, transfer, trading

        else:

            rewards = np.zeros(2)
            done = False
            info = None

            # get agents Q-values

            q_vals_a1 = self.agents[0].compute_q_values(observations[0])
            q_vals_a2 = self.agents[1].compute_q_values(observations[1])
            q_vals = [q_vals_a1, q_vals_a2]
            transfer = max(q_vals)

            # calculate compensation reward depending on Q-value

            # transfer = 0
            # pay_to = None

            # for i_agent, agent in enumerate(self.agents):
            #     if not greedy[i_agent]:
            #         transfer = np.maximum(np.max(q_vals[i_agent]), 0) * self.mark_up

            # storing actions for later decision
            # t_actions = []
            #
            # for t_step in range(self.n_trade_steps):
            #     for i_agent, agent in enumerate(self.agents):
            #         if greedy[i_agent]:
            #             t_actions.append(self.agents[i_agent].forward(observations[i_agent]))
            #         else:
            #             t_actions.append(np.argmin(q_vals[i_agent]))
            #             pay_to = i_agent

            observations, r, done, info = env.step(actions)
            observations = deepcopy(observations)
            return observations, rewards, done, info, transfer, trading

    def exchange_reward(self, env, rewards, suggested_steps, transfer):
        transfer_reward = transfer
        r = rewards

        suggested_action = []
        current_actions = []
        if len(env.actions_log) >= (env.nb_agents * env.trading_steps + env.nb_agents):
            for i_last in range(env.nb_agents):
                current_actions.append(suggested_steps(env.nb_agents * env.trading_steps + i_last))
            for i_suggested in range(env.nb_agents * env.trading_steps):
                suggested_action.append(suggested_steps(i_suggested))

            if current_actions[0] == 0:
                if suggested_action[0] == 1:
                    if current_actions[0][1] == suggested_action[0][3] and current_actions[0][0] == suggested_action[0][2]:
                        r[0] += transfer_reward
                if suggested_action[1] == 1:
                    if current_actions[0][1] == suggested_action[1][3] and current_actions[0][0] == suggested_action[1][2]:
                        r[0] += transfer_reward
            if current_actions[0] == 1:
                if suggested_action[0] == 0:
                    if current_actions[0][1] == suggested_action[0][3] and current_actions[0][0] == suggested_action[0][2]:
                        r[0] += transfer_reward
                if suggested_action[1] == 0:
                    if current_actions[0][1] == suggested_action[1][3] and current_actions[0][0] == suggested_action[1][2]:
                        r[0] += transfer_reward

        return r, transfer_reward


# test trading

def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    t = params.trading
    action_space = setup_action_space(params.trading_steps, params.trading_steps, None)
    
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
                       trading_steps=params.trading_steps,
                       trading_actions=action_space,
                       contracting=0,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks
                       )

    processor = env.SmartfactoryProcessor()

    trade = None

    if t != 0:
        trading_agents = []
        for i in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=len(action_space), processor=processor)
            agent.load_weights('experiments/20190923-10-58-52/run-0/contracting-0/dqn_weights-agent-{}.h5f'.format(i))
            # agent.load_weights('experiments/20191015-09-39-50/run-0/contracting-0/dqn_weights-agent-{}.h5f'.format(i))
            trading_agents.append(agent)
        trade = Trade(agent_1=trading_agents[0], agent_2=trading_agents[1], n_trade_steps=params.trading_steps,
                      mark_up=params.mark_up)

        agents = []
        for i_agent in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=env.nb_actions, processor=processor)
            agents.append(agent)
            agents[i_agent].load_weights(
                'experiments/20190923-10-58-52/run-0/contracting-{}/dqn_weights-agent-{}.h5f'.format(0, i_agent))
            # 'experiments/20191017-15-11-23/step-penalty-0.001/run-0.001/contracting-{}/dqn_weights-agent-{}.h5f'.format( 0, i_agent))

        combined_frames = []
        for i_episode in range(episodes):
            observations = env.reset()
            episode_rewards = np.zeros(params.nb_agents)
            accumulated_transfer = np.zeros(params.nb_agents)

            suggested_steps = []
            transfer = 0
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

                trading = False

                if trade is not None:

                    observations, r, done, info, transfer, trading = trade.trade_n_steps(env, observations, actions)

                    # list with agents and their suggested actions
                    suggested_steps = env.check_suggested_steps(trading_steps=params.trading_steps)
                else:
                    observations, r, done, info = env.step(actions)

                observations = deepcopy(observations)

                if trade is not None and not any([agent.done for agent in env.agents]):
                    r, act_transfer = trade.exchange_reward(env, r, suggested_steps, transfer)
                    accumulated_transfer += act_transfer
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

        #df.to_csv(os.path.join('test-values-trading-t-{}.csv'.format(0)))
        export_video('Smart-Factory-Trading.mp4', combined_frames, None)


if __name__ == '__main__':
    main()
