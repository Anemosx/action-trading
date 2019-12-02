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

    def __init__(self, valuation_nets, agents, n_trade_steps, mark_up, pay_up_front, trading_budget):
        self.agents = agents
        self.agent_count = len(agents)
        self.n_trade_steps = n_trade_steps
        self.mark_up = mark_up
        self.pay_up_front = pay_up_front
        self.valuation_nets = valuation_nets
        self.trading_budget = trading_budget

    def update_trading(self, rewards, episode_rewards, env, observations, suggested_steps, transfer):
        act_transfer = np.zeros(self.agent_count)
        current_actions = env.get_current_actions()
        new_trades = [0, 0]

        if self.n_trade_steps == 0:
            return rewards, suggested_steps, transfer, new_trades, act_transfer

        for i_agents in range(self.agent_count):
            agent_of_action = current_actions[i_agents][0]
            if len(suggested_steps[agent_of_action]) != 0:
                if suggested_steps[agent_of_action][0] == current_actions[i_agents][1][0] and \
                        suggested_steps[agent_of_action][1] == current_actions[i_agents][1][1]:
                    suggested_steps[agent_of_action] = suggested_steps[agent_of_action][2:]
                    if len(suggested_steps[agent_of_action]) == 0:
                        if self.pay_up_front:
                            new_trades[(agent_of_action + 1) % 2] += 1
                        else:
                            rewards, transfer, act_transfer, trade_success = self.pay_reward(agent_of_action, rewards, episode_rewards, transfer)
                            if trade_success:
                                new_trades[(agent_of_action + 1) % 2] += 1
                    else:
                        transfer[agent_of_action] += self.compensation_value(agent_of_action, suggested_steps[agent_of_action][:2], env.priorities, observations)
                else:
                    suggested_steps[agent_of_action] = []

        new_suggestions = [False, False]
        for i_agents in range(self.agent_count):
            if len(suggested_steps[i_agents]) == 0:
                new_suggestions[i_agents] = True
            else:
                new_suggestions[i_agents] = False

        for i_suggestion in range(len(new_suggestions)):
            if new_suggestions[i_suggestion]:
                copy_action_from = (i_suggestion + 1) % 2
                current_actions[copy_action_from][1] = current_actions[copy_action_from][1][2:]
                suggest = False
                for i in range(len(current_actions[copy_action_from][1])):
                    if current_actions[copy_action_from][1][i] != 0.0:
                        suggest = True
                if suggest:
                    suggested_steps[i_suggestion] = deepcopy(current_actions[copy_action_from][1])
                    transfer[i_suggestion] = self.compensation_value(i_suggestion, suggested_steps[i_suggestion][:2], env.priorities, observations)
                    if self.pay_up_front == 1:
                        rewards, transfer, act_transfer, trade_success = self.pay_reward(i_suggestion, rewards, episode_rewards, transfer)
                        transfer[i_suggestion] = 0

        return rewards, suggested_steps, transfer, new_trades, act_transfer

    def pay_reward(self, receiver, rewards, episode_rewards, transfer):
        trade_success = False
        act_transfer = [0, 0]
        payer = (receiver + 1) % 2

        if self.trading_budget[payer] > 0:
            if self.trading_budget[payer] - transfer[receiver] < 0:
                transfer[receiver] = deepcopy(self.trading_budget[payer])

            if episode_rewards[payer] - transfer[receiver] >= 0:
                rewards[payer] -= transfer[receiver]
                rewards[receiver] += transfer[receiver]
                self.trading_budget[payer] -= transfer[receiver]
                act_transfer[payer] += transfer[receiver]
                transfer[receiver] = 0
                trade_success = True

        return rewards, transfer, act_transfer, trade_success

    def check_actions(self, suggested_steps):
        tr_action_possibility = [False, False]
        if self.n_trade_steps == 0:
            return tr_action_possibility

        for i_agent in range(self.agent_count):
            if len(suggested_steps[(i_agent + 1) % 2]) == 0:
                tr_action_possibility[i_agent] = True

        return tr_action_possibility

    def compensation_value(self, receiver, action, priorities, observations):
        action_index = -1
        if action[1] == 1.0:  # up
            action_index = 0
        if action[1] == -1.0:  # down
            action_index = 1
        if action[0] == -1.0:  # left
            action_index = 2
        if action[0] == 1.0:  # right
            action_index = 3

        if priorities[receiver] == 0:
            # low priority
            if receiver == 0:
                highest_q_value = np.max(self.valuation_nets[0].compute_q_values(observations[0]))
                action_q_value = self.valuation_nets[0].compute_q_values(observations[0])[action_index]
            else:
                highest_q_value = np.max(self.valuation_nets[0].compute_q_values(observations[1]))
                action_q_value = self.valuation_nets[0].compute_q_values(observations[1])[action_index]
        else:
            # high priority
            if receiver == 0:
                highest_q_value = np.max(self.valuation_nets[1].compute_q_values(observations[0]))
                action_q_value = self.valuation_nets[1].compute_q_values(observations[0])[action_index]
            else:
                highest_q_value = np.max(self.valuation_nets[1].compute_q_values(observations[1]))
                action_q_value = self.valuation_nets[1].compute_q_values(observations[1])[action_index]

        compensation = (highest_q_value - action_q_value) * self.mark_up

        return compensation


def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    episodes = 1
    episode_steps = 100

    ep_columns = ['episode', 'trading', 'reward', 'number_trades', 'episode_steps', 'episode_trades']
    for i_ag in range(params.nb_agents):
        ag_columns = ['reward_a{}'.format(i_ag),
                      'accumulated_transfer_a{}'.format(i_ag),
                      'trades_a-{}'.format(i_ag)]
        ep_columns += ag_columns
    df = pd.DataFrame(columns=ep_columns)

    action_space = setup_action_space(params.trading_steps, params.trading_steps, None)

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       trading=params.trading,
                       trading_steps=params.trading_steps,
                       trading_actions=action_space,
                       trading_signals=params.trading_signals,
                       priorities=params.priorities,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks)

    processor = env.SmartfactoryProcessor()

    trading_agents = []
    agents = []
    for _ in range(params.nb_agents):
        if params.trading == 2:
            agent = build_agent(params=params, nb_actions=env.nb_actions - 4, processor=processor)
            trading_agents.append(agent)

            no_tr_agent = build_agent(params=params, nb_actions=4, processor=processor)
            agents.append(no_tr_agent)
        else:
            agent = build_agent(params=params, nb_actions=env.nb_actions, processor=processor)
            trading_agents.append(agent)

    valuation_low_priority = build_agent(params=params, nb_actions=4, processor=processor)
    valuation_low_priority.load_weights('experiments/20191106-11-32-13/run-0/contracting-0/dqn_weights-agent-0.h5f')

    valuation_high_priority = build_agent(params=params, nb_actions=4, processor=processor)
    valuation_high_priority.load_weights('experiments/20191106-11-30-59/run-0/contracting-0/dqn_weights-agent-0.h5f')

    valuation_nets = [valuation_low_priority, valuation_high_priority]

    trade = Trade(valuation_nets=valuation_nets,
                  agents=trading_agents,
                  n_trade_steps=params.trading_steps,
                  mark_up=params.mark_up,
                  pay_up_front=params.pay_up_front,
                  trading_budget=params.trading_budget)

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
        trade.trading_budget = params.trading_budget

        for i in range(params.nb_agents):
            if trade.agents[i].processor is not None:
                observations[i] = trade.agents[i].processor.process_observation(observations[i])
                q_vals[i] = trade.agents[i].compute_q_values(observations[i])

        combined_frames = drawing_util.render_combined_frames(combined_frames, env, [0, 0], 0, [0, 0])

        for i_step in range(episode_steps):
            actions = []
            for i_ag in range(params.nb_agents):
                if not env.agents[i_ag].done:
                    if params.trading == 2:
                        tr_checks = trade.check_actions(suggested_steps)
                        if tr_checks[i_ag]:
                            actions.append(trade.agents[i_ag].forward(observations[i_ag]) + 4)
                        else:
                            actions.append(agents[i_ag].forward(observations[i_ag]))
                    else:
                        actions.append(trade.agents[i_ag].forward(observations[i_ag]))
                    if trade.agents[i_ag].processor is not None:
                        actions[i_ag] = trade.agents[i_ag].processor.process_action(actions[i_ag])
                else:
                    actions.append(0)

            observations, r, done, info = env.step(actions)
            observations = deepcopy(observations)

            if trade.n_trade_steps > 0 and not done:
                for i in range(params.nb_agents):
                    if trade.agents[i].processor is not None:
                        observations[i] = trade.agents[i].processor.process_observation(observations[i])

                r, suggested_steps, transfer, new_trades, act_transfer = trade.update_trading(r, episode_rewards, env, observations, suggested_steps, transfer)
                observations = env.update_trade_colors(suggested_steps)
            else:
                new_trades = [0, 0]
                act_transfer = [0, 0]

            for i in range(params.nb_agents):
                if trade.agents[i].processor is not None:
                    observations[i] = trade.agents[i].processor.process_observation(observations[i])
                    q_vals[i] = trade.agents[i].compute_q_values(observations[i])

            accumulated_transfer += act_transfer
            trade_count += new_trades
            episode_rewards += r

            if not env.agents[0].done and not env.agents[1].done:
                info_trade = 0
                for i in range(len(new_trades)):
                    if new_trades[i] != 0:
                        info_trade = 1
                for i in range(3):
                    combined_frames = drawing_util.render_combined_frames(combined_frames, env, r, info_trade, actions, q_vals)

            if done:
                ep_stats = [i_episode, (trade is not None), np.sum(episode_rewards), 0, episode_steps, np.sum(trade_count)]
                for i_ag in range(len(agents)):
                    ag_stats = [episode_rewards[i_ag], accumulated_transfer[i_ag], trade_count[i_ag]]
                    ep_stats += ag_stats
                df.loc[i_episode] = ep_stats
                break

        print("Rewards: ", episode_rewards)
        print("Trades: ", trade_count)

        # df.to_csv(os.path.join('test-values-trading-t-{}.csv'.format(0)))
        export_video('Smart-Factory-Trading.mp4', combined_frames, None)


if __name__ == '__main__':
    main()
