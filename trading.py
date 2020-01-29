import numpy as np
import os
from copy import deepcopy
import envs.smartfactory
from common_utils.utils import export_video
from dotmap import DotMap
import json
import common_utils.drawing_util as drawing_util
from agents.pytorch_agents import make_dqn_agent


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


def eval_mode_setup(params):
    mode_str = ""
    eval_list = [-1]

    if params.eval_mode == 0:
        eval_list = params.eval_trading_steps
        mode_str = "trading steps"
    if params.eval_mode == 1:
        eval_list = params.eval_trading_budget
        mode_str = "trading budget"
    if params.eval_mode == 2:
        eval_list = params.eval_mark_up
        mode_str = "mark up"
    if params.eval_mode == -1:
        mode_str = "val net"
    if params.partial_pay:
        params.eval_mode = 0
        params.trading_steps = 10
        eval_list = [10]
        mode_str = "partial pay"

    return mode_str, eval_list


class Trade:

    def __init__(self, env, params, agents, suggestion_agents):
        self.trading_mode = params.trading_mode
        self.agents = agents
        self.suggestion_agents = suggestion_agents
        self.trading_steps = params.trading_steps
        self.mark_up = params.mark_up
        self.pay_up_front = params.pay_up_front
        self.trading_budget = params.trading_budget
        self.gamma = params.gamma
        self.suggested_steps = [[], []]
        self.transfer = np.zeros(len(agents))
        self.step_transfer = [[], []]
        self.trades = np.zeros(shape=(len(self.agents), self.trading_steps+1), dtype=int)
        self.partial_pay = params.partial_pay

        if self.trading_steps > 0:
            # dedicated observation for valuation
            observation_valuation_shape = list(env.observation_valuation_space.shape)

            valuation_low_priority = make_dqn_agent(params, observation_valuation_shape, 4)
            valuation_low_priority.load_weights('C:/Users/' + os.environ.get(
                'USERNAME') + '/contracting-agents/valuation_nets/rw {} pen {}/low_priority.pth'.format(params.rewards, params.step_penalties))
            valuation_high_priority = make_dqn_agent(params, observation_valuation_shape, 4)
            valuation_high_priority.load_weights('C:/Users/' + os.environ.get(
                'USERNAME') + '/contracting-agents/valuation_nets/rw {} pen {}/high_priority.pth'.format(params.rewards, params.step_penalties))

            self.valuation_nets = [valuation_low_priority, valuation_high_priority]

    def trading_step(self, episode_rewards, env, actions):

        observations, rewards, joint_done, info = env.step(actions)

        new_trades = np.zeros(len(self.agents), dtype=int)
        act_transfer = np.zeros(len(self.agents))

        if self.trading_steps > 0:
            current_actions = env.get_current_actions()

            rewards, act_transfer, new_trades = self.action_comparison(current_actions, rewards, episode_rewards, new_trades, act_transfer)

            if not joint_done.__contains__(True):

                rewards, act_transfer = self.add_suggestions(current_actions, env, episode_rewards, rewards, joint_done, act_transfer)

            observations = env.update_trade_colors(self.suggested_steps)

        return rewards, observations, joint_done, new_trades, act_transfer

    def action_comparison(self, current_actions, rewards, episode_rewards, new_trades, act_transfer):
        for i_agents in range(len(self.agents)):
            agent_of_action = current_actions[i_agents][0]
            other_agent = (agent_of_action + 1) % 2
            if len(self.suggested_steps[agent_of_action]) > 0:
                if self.suggested_steps[agent_of_action][0] == current_actions[i_agents][1][0] and \
                        self.suggested_steps[agent_of_action][1] == current_actions[i_agents][1][1]:
                    self.suggested_steps[agent_of_action] = self.suggested_steps[agent_of_action][2:]
                    if len(self.suggested_steps[agent_of_action]) == 0:
                        if self.pay_up_front:
                            new_trades[other_agent] += 1
                        else:
                            rewards, act_transfer_pay, trade_success = self.pay_reward(agent_of_action, rewards, episode_rewards)
                            if trade_success:
                                act_transfer[other_agent] += act_transfer_pay[other_agent]
                                new_trades[other_agent] += 1
                                self.trades[other_agent][self.trading_steps] += 1
                else:
                    if self.pay_up_front:
                        rewards[other_agent] += self.transfer[agent_of_action]
                        rewards[agent_of_action] -= self.transfer[agent_of_action]
                        act_transfer[other_agent] -= self.transfer[agent_of_action]
                        self.transfer[agent_of_action] = 0
                    elif self.partial_pay:
                        nb_followed = int(self.trading_steps - len(self.suggested_steps[agent_of_action]) / 2)
                        if nb_followed > 0:
                            self.transfer[agent_of_action] = np.sum(self.step_transfer[agent_of_action][:nb_followed])
                            rewards, act_transfer_pay, trade_success = self.pay_reward(agent_of_action, rewards, episode_rewards)
                            if trade_success:
                                act_transfer[other_agent] += act_transfer_pay[other_agent]
                                self.trades[other_agent][nb_followed] += 1
                    self.suggested_steps[agent_of_action] = []

        return rewards, act_transfer, new_trades

    def add_suggestions(self, current_actions, env, episode_rewards, rewards, joint_done, act_transfer):

        for i_agent in range(len(self.agents)):
            if len(self.suggested_steps[i_agent]) == 0:
                other_agent = (i_agent + 1) % 2

                # normal suggestion adding
                if self.trading_mode == 0:

                    current_actions[other_agent][1] = current_actions[other_agent][1][2:]
                    suggest = False
                    for i in range(len(current_actions[other_agent][1])):
                        if current_actions[other_agent][1][i] != 0.0:
                            suggest = True
                    if suggest:
                        self.suggested_steps[i_agent] = deepcopy(current_actions[other_agent][1])
                        self.transfer[i_agent] = self.compensation_n_steps(i_agent, env)

                        if self.pay_up_front:
                            rewards, act_transfer_pay, trade_success = self.pay_reward(i_agent, rewards, episode_rewards)
                            if not trade_success:
                                self.suggested_steps[i_agent] = []
                            else:
                                act_transfer[other_agent] += act_transfer_pay[other_agent]
                                self.transfer[i_agent] = act_transfer_pay[other_agent]

                # suggestion adding with 1, separated suggestion agent 2, extra observation channel
                elif self.trading_mode == 1 or self.trading_mode == 2:

                    if current_actions[other_agent][1][2]:
                        if self.trading_mode == 2:
                            env.set_missing_suggestion(other_agent, 1)

                        suggestions_observations = env.get_observation_trade(other_agent)
                        for i_trading_steps in range(self.trading_steps):
                            action = self.suggestion_agents[other_agent].policy(suggestions_observations)

                            if action % 4 == 0:  # up
                                self.suggested_steps[i_agent].extend([0.0, 1.0])
                            if action % 4 == 1:  # down
                                self.suggested_steps[i_agent].extend([0.0, -1.0])
                            if action % 4 == 2:  # left
                                self.suggested_steps[i_agent].extend([-1.0, 0.0])
                            if action % 4 == 3:  # right
                                self.suggested_steps[i_agent].extend([1.0, 0.0])

                            env.set_suggestions(self.suggested_steps)

                            suggestions_observations_after = env.get_observation_trade(other_agent)

                            self.suggestion_agents[other_agent].save(suggestions_observations,
                                                                     action,
                                                                     suggestions_observations_after,
                                                                     0,
                                                                     joint_done[other_agent])

                            suggestions_observations = suggestions_observations_after

                        if self.trading_mode == 2:
                            env.set_missing_suggestion(other_agent, 0)

                        self.transfer[i_agent] = self.compensation_n_steps(i_agent, env)

                        if self.pay_up_front:
                            rewards, act_transfer_pay, trade_success = self.pay_reward(i_agent, rewards,
                                                                                       episode_rewards)
                            if not trade_success:
                                self.suggested_steps[i_agent] = []
                            else:
                                act_transfer[other_agent] += act_transfer_pay[other_agent]
                                self.transfer[i_agent] = act_transfer_pay[other_agent]

        if self.trading_mode == 1:
            for i_agent in range(len(self.agents)):
                if not joint_done[i_agent]:
                    self.suggestion_agents[i_agent].train()

        return rewards, act_transfer

    def pay_reward(self, receiver, rewards, episode_rewards):
        trade_success = False
        act_transfer = np.zeros(len(self.agents))
        payer = (receiver + 1) % 2

        if episode_rewards[payer] <= 0 or self.transfer[receiver] == 0:
            self.transfer[receiver] = 0
            return rewards, act_transfer, trade_success

        self.transfer[receiver] = self.transfer[receiver] * self.mark_up

        if self.trading_budget[payer] > 0:
            if self.trading_budget[payer] - self.transfer[receiver] < 0:
                self.transfer[receiver] = deepcopy(self.trading_budget[payer])

            if episode_rewards[payer] - self.transfer[receiver] >= 0:
                rewards[payer] -= self.transfer[receiver]
                rewards[receiver] += self.transfer[receiver]
                self.trading_budget[payer] -= self.transfer[receiver]
                act_transfer[payer] += self.transfer[receiver]
                trade_success = True

        self.transfer[receiver] = 0

        return rewards, act_transfer, trade_success

    def compensation_value(self, receiver, suggested_action, env):
        action_index = -1
        if suggested_action[1] == 1.0:  # up
            action_index = 0
        if suggested_action[1] == -1.0:  # down
            action_index = 1
        if suggested_action[0] == -1.0:  # left
            action_index = 2
        if suggested_action[0] == 1.0:  # right
            action_index = 3

        valuation_q_vals = self.valuation_nets[env.priorities[receiver]].compute_q_values(env.get_valuation_observation(receiver))[0]
        compensation = (np.max(valuation_q_vals) - valuation_q_vals[action_index]) / self.gamma

        return compensation

    def compensation_n_steps(self, receiver, env):
        sf_values = env.store_values()
        remaining_suggestion = deepcopy(self.suggested_steps)
        self.step_transfer[receiver] = []
        self.step_transfer[receiver].append(self.compensation_value(receiver, remaining_suggestion[receiver][:2], env))
        observations = env.update_trade_colors(remaining_suggestion)
        joint_done = [False, False]
        for i_step in range(self.trading_steps - 1):
            actions = []
            for agent_index in [0, 1]:
                if not joint_done[agent_index]:
                    if agent_index != receiver:
                        action = self.agents[agent_index].policy(observations[agent_index])
                    else:
                        action = -1
                        if remaining_suggestion[receiver][1] == 1.0:  # up
                            action = 0
                        if remaining_suggestion[receiver][1] == -1.0:  # down
                            action = 1
                        if remaining_suggestion[receiver][0] == -1.0:  # left
                            action = 2
                        if remaining_suggestion[receiver][0] == 1.0:  # right
                            action = 3
                else:
                    action = np.random.randint(0, 4)
                actions.append(action)
            new_observations, rewards, joint_done, info = env.step(actions)
            observations = new_observations
            remaining_suggestion[receiver] = remaining_suggestion[receiver][2:]
            self.step_transfer[receiver].append(self.compensation_value(receiver, remaining_suggestion[receiver][:2], env))

        env.load_values(sf_values)

        return np.sum(self.step_transfer[receiver])


def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    name_dir = 'trading steps - tr mode 2 - 20200113-18-44-21'
    agent_dir = 'trading steps 1'

    trained_agents = True
    episodes = 1
    episode_steps = 100

    env = envs.smartfactory.make_smart_factory(params)
    observation_shape = list(env.observation_space.shape)
    number_of_actions = env.action_space.n

    agents = []
    suggestion_agents = []
    for i_ag in range(params.nb_agents):
        ag = make_dqn_agent(params, observation_shape, number_of_actions)
        if trained_agents:
            ag.load_weights(os.path.join(os.getcwd(), 'exp-trading/{}/{}/weights-{}.pth'.format(name_dir, agent_dir, i_ag)))
            ag.epsilon = 0.01
        agents.append(ag)
    if params.trading_mode == 1:
        for i_ag in range(params.nb_agents):
            suggestion_ag = make_dqn_agent(params, observation_shape, 4)
            if trained_agents:
                suggestion_ag.load_weights(os.path.join(os.getcwd(), 'exp-trading/{}/{}/weights-sugg-{}.pth'.format(name_dir, agent_dir, i_ag)))
                suggestion_ag.epsilon = 0.01
            suggestion_agents.append(suggestion_ag)
    if params.trading_mode == 2:
        suggestion_agents = agents

    trade = Trade(env=env, params=params, agents=agents, suggestion_agents=suggestion_agents)

    env.set_render_mode(True)
    combined_frames = []

    for episode in range(0, episodes):
        observations = env.reset()
        done = False
        current_step = 0
        agent_indices = list(range(0, len(env.agents)))
        episode_return = np.zeros(len(env.agents))
        joint_done = [False, False]
        trade.trading_budget = deepcopy(params.trading_budget)
        trade_count = np.zeros(len(agents))
        trade.trades = np.zeros(shape=(len(trade.agents), trade.trading_steps+1))
        accumulated_transfer = np.zeros(len(agents))
        combined_frames = drawing_util.render_combined_frames(combined_frames, env, [0, 0], 0, [0, 0])

        while not done:

            actions = []
            for agent_index in agent_indices:
                if not joint_done[agent_index]:
                    action = agents[agent_index].policy(observations[agent_index])
                else:
                    action = np.random.randint(0, 4)
                actions.append(action)

            joint_reward, next_observations, joint_done, new_trades, act_transfer = trade.trading_step(episode_return, env, actions)

            observations = next_observations

            current_step += 1
            for i in range(len(trade.agents)):
                episode_return[i] += joint_reward[i]
                trade_count[i] += new_trades[i]
                accumulated_transfer[i] += act_transfer[i]

            info_trade = 0
            if trade.trading_steps > 0:
                for i in range(len(trade.agents)):
                    if new_trades[i] != 0:
                        info_trade = 1
            for i in range(3):
                combined_frames = drawing_util.render_combined_frames(combined_frames, env, episode_return, info_trade, actions, [0, 0])

            if not params.done_mode:
                done = all(done is True for done in joint_done) or current_step == episode_steps
            else:
                done = joint_done.__contains__(True) or current_step == episode_steps

        print("steps: {} | rewards: {} | complete trades: {} | transfer: {}".format(current_step, episode_return, trade_count, accumulated_transfer))

        print("trades: ", trade.trades)

    export_video('Smart-Factory-Trading.mp4', combined_frames, None)


if __name__ == '__main__':
    main()
