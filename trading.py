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
        # if trading_steps != 0:
        #     no_tr_action_space = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
        #     for i in range(len(no_tr_action_space)):
        #         no_tr_action_space[i] = no_tr_action_space[i] + [0.0, 0.0] * trading_steps
        #     tr_action_space = no_tr_action_space + tr_action_space
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

    def __init__(self, valuation_nets, agent_1, agent_2, n_trade_steps, mark_up):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.agents = [agent_1, agent_2]
        self.agent_count = 2
        self.n_trade_steps = n_trade_steps
        self.mark_up = mark_up
        self.valuation_nets = valuation_nets

    # trading part of step
    #   set flag to adding suggestion actions
    #   step
    #   load current action
    #   if current action == suggestion action
    #       del suggestion action
    #       if suggestion actions == None
    #           pay reward
    #   else
    #       del suggestion actions
    #   set new suggestion actions depending on flags

    def update_trading(self, r, env, observations, suggested_steps, transfer):
        act_transfer = np.zeros(self.agent_count)
        current_actions = deepcopy(env.get_current_actions())
        rewards = deepcopy(r)
        new_trades = [0, 0]

        if self.n_trade_steps == 0:
            return rewards, suggested_steps, transfer, new_trades, act_transfer

        for i_agents in range(self.agent_count):
            agent_of_action = current_actions[i_agents][0]
            if len(suggested_steps[agent_of_action]) != 0:
                if suggested_steps[agent_of_action][0] == current_actions[i_agents][1][0] and \
                        suggested_steps[agent_of_action][1] == current_actions[i_agents][1][1]:
                    del suggested_steps[agent_of_action][0]
                    del suggested_steps[agent_of_action][0]

                    if len(suggested_steps[agent_of_action]) == 0:
                        rewards, transfer, act_agent_transfer, trade_success = self.pay_reward((agent_of_action + 1) % 2, agent_of_action, rewards,
                                                                          transfer)
                        act_transfer[agent_of_action] += act_agent_transfer
                        if trade_success:
                            new_trades[(agent_of_action + 1) % 2] += 1
                        # act_transfer[agent_of_action] += transfer[agent_of_action]
                else:
                    suggested_steps[agent_of_action] = []

        new_trade = [False, False]
        for i_agents in range(self.agent_count):
            if len(suggested_steps[i_agents]) == 0:
                new_trade[i_agents] = True
            else:
                new_trade[i_agents] = False

        for i_trades in range(len(new_trade)):
            if new_trade[i_trades]:
                copy_action_from = (i_trades + 1) % 2
                del current_actions[copy_action_from][1][0]
                del current_actions[copy_action_from][1][0]
                if current_actions[copy_action_from][1] != [0.0, 0.0]:
                    suggested_steps[i_trades] = deepcopy(current_actions[copy_action_from][1])

                    # if copy_action_from == 0:
                    #     q_val = q_vals[1]
                    # else:
                    #     q_val = q_vals[0]
                    # transfer[i_trades] = np.max(q_val) * 1.01
                    # transfer[i_trades] = 0.01

                    action = [suggested_steps[i_trades][0], suggested_steps[i_trades][1]]
                    transfer[i_trades] = self.compensation_value(i_trades, action, env.priorities, observations)
                    new_trade[i_trades] = False

        return rewards, suggested_steps, transfer, new_trades, act_transfer

    def pay_reward(self, payer, receiver, rewards, transfer_value):
        new_rewards = deepcopy(rewards)
        new_transfer = [0, 0]

        if new_rewards[payer] - transfer_value[receiver] >= 0:
            new_rewards[payer] -= transfer_value[receiver]
            new_rewards[receiver] += transfer_value[receiver]
            act_transfer = transfer_value[receiver]

            new_transfer[payer] = transfer_value[payer]
            new_transfer[receiver] = 0
            trade_success = True
        else:
            new_transfer = transfer_value
            act_transfer = 0
            trade_success = False

        return new_rewards, new_transfer, act_transfer, trade_success

    def check_actions(self, suggested_steps):
        tr_action_possibility = [False, False]
        if self.n_trade_steps == 0:
            return tr_action_possibility
        for i_agent in range(self.agent_count):
            if not suggested_steps[(i_agent + 1) % 2]:
                tr_action_possibility[i_agent] = True
        return tr_action_possibility

    def compensation_value(self, receiver, action, priorities, observations):
        q_val_index = -1
        if action[1] == 1.0:  # up
            q_val_index = 0
        if action[1] == -1.0:  # down
            q_val_index = 1
        if action[0] == -1.0:  # left
            q_val_index = 2
        if action[0] == 1.0:  # right
            q_val_index = 3

        if action == [0.0, 0.0] or q_val_index == -1:
            return 0

        observations = deepcopy(observations)
        agent_priority = deepcopy(priorities[receiver])
        if agent_priority == 0:
            # low priority
            if receiver == 0:
                highest_q_value = np.max(self.valuation_nets[0].compute_q_values(observations[0]))
                action_q_value = self.valuation_nets[0].compute_q_values(observations[0])[q_val_index]
            else:
                highest_q_value = np.max(self.valuation_nets[0].compute_q_values(observations[1]))
                action_q_value = self.valuation_nets[0].compute_q_values(observations[1])[q_val_index]
        else:
            # high priority
            if receiver == 0:
                highest_q_value = np.max(self.valuation_nets[1].compute_q_values(observations[0]))
                action_q_value = self.valuation_nets[1].compute_q_values(observations[0])[q_val_index]
            else:
                highest_q_value = np.max(self.valuation_nets[1].compute_q_values(observations[1]))
                action_q_value = self.valuation_nets[1].compute_q_values(observations[1])[q_val_index]

        compensation = (highest_q_value - action_q_value) * self.mark_up
        return compensation


# trading

def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    if params.trading_steps != 0 and params.trading != 0:
        action_space = setup_action_space(params.trading_steps, params.trading_steps, None)
    else:
        action_space = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]

    policy_random = False
    episodes = 1
    episode_steps = 100

    ep_columns = ['episode', 'trading', 'reward', 'number_trades', 'episode_steps', 'episode_trades']
    for i_ag in range(params.nb_agents):
        ag_columns = ['reward_a{}'.format(i_ag),
                      'accumulated_transfer_a{}'.format(i_ag),
                      'trades_a-{}'.format(i_ag)]
        ep_columns += ag_columns
    df = pd.DataFrame(columns=ep_columns)

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       trading=params.trading,
                       trading_steps=params.trading_steps,
                       trading_actions=action_space,
                       contracting=0,
                       priorities=params.priorities,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks
                       )

    processor = env.SmartfactoryProcessor()
    if params.trading != 0:
        trading_agents = []
        for i in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=len(action_space), processor=processor)
            trading_agents.append(agent)

        valuation_low_priority = build_agent(params=params, nb_actions=4, processor=processor)
        valuation_low_priority.load_weights('experiments/20191106-11-32-13/run-0/contracting-0/dqn_weights-agent-0.h5f')

        valuation_high_priority = build_agent(params=params, nb_actions=4, processor=processor)
        valuation_high_priority.load_weights('experiments/20191106-11-30-59/run-0/contracting-0/dqn_weights-agent-0.h5f')

        valuation_nets = [valuation_low_priority, valuation_high_priority]

        trade = Trade(valuation_nets=valuation_nets, agent_1=trading_agents[0], agent_2=trading_agents[1], n_trade_steps=params.trading_steps,
                      mark_up=params.mark_up)
    else:
        trade = None

    agents = []
    for i_agent in range(params.nb_agents):
        agent = build_agent(params=params, nb_actions=4, processor=processor)
        agents.append(agent)

    combined_frames = []

    for i_episode in range(episodes):
        observations = env.reset()
        observations = deepcopy(observations)
        episode_rewards = np.zeros(params.nb_agents)
        accumulated_transfer = np.zeros(params.nb_agents)
        transfer = np.zeros(params.nb_agents)
        q_vals = [[], []]
        trade_count = np.zeros(params.nb_agents)
        suggested_steps = [[], []]

        if trade is not None:
            for i in range(2):
                if trade.agents[i].processor is not None:
                    observations[i] = trade.agents[i].processor.process_observation(observations[i])
            for i in range(2):
                q_val = trade.agents[i].compute_q_values(observations[i])
                q_vals[i] = q_val
        else:
            for i in range(2):
                if agents[i].processor is not None:
                    observations[i] = agents[i].processor.process_observation(observations[i])
            for i in range(2):
                q_val = agents[i].compute_q_values(observations[i])
                q_vals[i] = q_val

        combined_frames = drawing_util.render_combined_frames(combined_frames, env, [0, 0], params.trading, [0, 0])

        for i_step in range(episode_steps):
            actions = []
            for i_ag in range(params.nb_agents):
                if not env.agents[i_ag].done:
                    if not policy_random:
                        actions.append(trade.agents[i_ag].forward(observations[i_ag]))
                        # if trade.check_actions(suggested_steps)[i_ag]:
                        #     actions.append(trade.agents[i_ag].forward(observations[i_ag])+4)
                        # else:
                        #     actions.append(agents[i_ag].forward(observations[i_ag]))
                    else:
                        actions.append(np.random.randint(0, env.nb_actions))
                else:
                    actions.append(0)

            observations, r, done, info = env.step(actions)
            observations = deepcopy(observations)

            if trade is not None and not any([agent.done for agent in env.agents]):
                for i in range(2):
                    if trade.agents[i].processor is not None:
                        observations[i] = trade.agents[i].processor.process_observation(observations[i])
                r, suggested_steps, transfer, new_trades, act_transfer = trade.update_trading(r, env, observations, suggested_steps, transfer)

                observations = env.update_trade_colors(suggested_steps)

            if trade is not None:
                for i in range(2):
                    if trade.agents[i].processor is not None:
                        observations[i] = trade.agents[i].processor.process_observation(observations[i])
                q_vals_a1 = trade.agents[0].compute_q_values(observations[0])
                q_vals_a2 = trade.agents[1].compute_q_values(observations[1])
            else:
                for i in range(2):
                    if agents[i].processor is not None:
                        observations[i] = agents[i].processor.process_observation(observations[i])
                q_vals_a1 = agents[0].compute_q_values(observations[0])
                q_vals_a2 = agents[1].compute_q_values(observations[1])
            q_vals = [q_vals_a1, q_vals_a2]

            accumulated_transfer += act_transfer
            trade_count += new_trades
            episode_rewards += r

            combined_frames = drawing_util.render_combined_frames(combined_frames, env, r, params.trading, actions, q_vals)

            if done:
                ep_stats = [i_episode, (trade is not None), np.sum(episode_rewards), 0, episode_steps, np.sum(trade_count)]
                for i_ag in range(len(agents)):
                    ag_stats = [episode_rewards[i_ag], accumulated_transfer[i_ag], trade_count[i_ag]]
                    ep_stats += ag_stats
                df.loc[i_episode] = ep_stats
                break

        print("Rewards: \n", episode_rewards)
        print("Trades: \n", trade_count)

        # df.to_csv(os.path.join('test-values-trading-t-{}.csv'.format(0)))
        export_video('Smart-Factory-Trading.mp4', combined_frames, None)


if __name__ == '__main__':
    main()
