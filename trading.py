import numpy as np
import agents.pytorch_agents as pta
from copy import deepcopy
from envs.smartfactory import Smartfactory
from common_utils.utils import export_video
from dotmap import DotMap
import json
import common_utils.drawing_util as drawing_util


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

    def __init__(self, valuation_nets, agents, trading, n_trade_steps, mark_up, gamma, pay_up_front, trading_budget):
        self.agents = agents
        self.agent_count = len(agents)
        self.trading = trading
        self.n_trade_steps = n_trade_steps
        self.mark_up = mark_up
        self.pay_up_front = pay_up_front
        self.valuation_nets = valuation_nets
        self.trading_budget = trading_budget
        self.gamma = gamma
        self.suggested_steps = [[], []]
        self.transfer = np.zeros(len(agents))

    def trading_step(self, episode_rewards, env, actions):

        observations, rewards, joint_done, info = env.step(actions)

        if self.trading > 0 and self.n_trade_steps > 0:

            for i_agent in range(self.agent_count):
                info['new_trades_{}'.format(i_agent)] = 0
                info['act_transfer_{}'.format(i_agent)] = 0
            current_actions = env.get_current_actions()
            old_suggested_steps = deepcopy(self.suggested_steps)

            for i_agents in range(self.agent_count):
                agent_of_action = current_actions[i_agents][0]
                other_agent = (agent_of_action + 1) % 2
                if len(self.suggested_steps[agent_of_action]) != 0:
                    if self.suggested_steps[agent_of_action][0] == current_actions[i_agents][1][0] and \
                            self.suggested_steps[agent_of_action][1] == current_actions[i_agents][1][1]:
                        self.suggested_steps[agent_of_action] = self.suggested_steps[agent_of_action][2:]
                        if len(self.suggested_steps[agent_of_action]) == 0:
                            if self.pay_up_front:
                                info['new_trades_{}'.format(other_agent)] += 1
                            else:
                                rewards, act_transfer, trade_success = self.pay_reward(agent_of_action, rewards,
                                                                                           episode_rewards)
                                info['act_transfer_{}'.format(other_agent)] += act_transfer[other_agent]
                                if trade_success:
                                    info['new_trades_{}'.format(other_agent)] += 1
                        else:
                            self.transfer[agent_of_action] += self.compensation_value(agent_of_action, self.suggested_steps[agent_of_action][:2], env.priorities, observations)
                    else:
                        self.suggested_steps[agent_of_action] = []

            new_suggestions = [False, False]
            for i_agents in range(self.agent_count):
                if len(self.suggested_steps[i_agents]) == 0:
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
                        self.suggested_steps[i_suggestion] = deepcopy(current_actions[copy_action_from][1])
                        self.transfer[i_suggestion] = self.compensation_value(i_suggestion,
                                                                              self.suggested_steps[i_suggestion][:2],
                                                                              env.priorities, observations)
                        if self.pay_up_front == 1:
                            rewards, act_transfer, trade_success = self.pay_reward(i_suggestion, rewards,
                                                                                       episode_rewards)
                            info['act_transfer_{}'.format(copy_action_from)] += act_transfer[copy_action_from]
                            self.transfer[i_suggestion] = 0

            if not (old_suggested_steps == self.suggested_steps):
                observations = env.update_trade_colors(self.suggested_steps)

        return rewards, observations, joint_done, info

    def pay_reward(self, receiver, rewards, episode_rewards):
        trade_success = False
        act_transfer = [0, 0]
        payer = (receiver + 1) % 2
        self.transfer[receiver] = self.transfer[receiver] * self.mark_up

        if self.trading_budget[payer] > 0:
            if self.trading_budget[payer] - self.transfer[receiver] < 0:
                self.transfer[receiver] = deepcopy(self.trading_budget[payer])

            if episode_rewards[payer] - self.transfer[receiver] >= 0:
                rewards[payer] -= self.transfer[receiver]
                rewards[receiver] += self.transfer[receiver]
                self.trading_budget[payer] -= self.transfer[receiver]
                act_transfer[payer] += self.transfer[receiver]
                self.transfer[receiver] = 0
                trade_success = True

        return rewards, act_transfer, trade_success

    def check_actions(self, suggested_steps):
        tr_action_possibility = [False, False]
        if self.trading == 0 or self.n_trade_steps == 0:
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

        valuation_q_vals = self.valuation_nets[priorities[receiver]].compute_q_values(observations[receiver])[0]
        compensation = (np.max(valuation_q_vals) - valuation_q_vals[action_index]) / self.gamma

        return compensation


def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    episodes = 1
    episode_steps = 100

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
                       nb_steps_machine_inactive=params.nb_steps_machine_inactive,
                       nb_tasks=params.nb_tasks,
                       observation=2
                       )

    observation_shape = list(env.observation_space.shape)
    number_of_actions = env.action_space.n

    agents = []
    no_tr_agents = []
    if params.trading == 2:
        for i_ag in range(params.nb_agents):
            ag = pta.DqnAgent(
                observation_shape=observation_shape,
                number_of_actions=4,
                gamma=0.95,
                epsilon_decay=0.00002,
                epsilon_min=0.0,
                mini_batch_size=64,
                warm_up_duration=1000,
                buffer_capacity=20000,
                target_update_period=2000,
                seed=1337)
            no_tr_agents.append(ag)

        for i_ag in range(params.nb_agents):
            ag = pta.DqnAgent(
                observation_shape=observation_shape,
                number_of_actions=number_of_actions - 4,
                gamma=0.95,
                epsilon_decay=0.00002,
                epsilon_min=0.0,
                mini_batch_size=64,
                warm_up_duration=1000,
                buffer_capacity=20000,
                target_update_period=2000,
                seed=1337)
            agents.append(ag)
    else:
        for i_ag in range(params.nb_agents):
            ag = pta.DqnAgent(
                observation_shape=observation_shape,
                number_of_actions=number_of_actions,
                gamma=0.95,
                epsilon_decay=0.00002,
                epsilon_min=0.0,
                mini_batch_size=64,
                warm_up_duration=1000,
                buffer_capacity=20000,
                target_update_period=2000,
                seed=1337)
            agents.append(ag)

    valuation_low_priority = pta.DqnAgent(
        observation_shape=observation_shape,
        number_of_actions=4,
        gamma=0.95,
        epsilon_decay=0.00002,
        epsilon_min=0.0,
        mini_batch_size=64,
        warm_up_duration=1000,
        buffer_capacity=20000,
        target_update_period=2000,
        seed=1337)
    valuation_low_priority.epsilon = 0.01
    valuation_low_priority.load_weights('valuation_nets/low_priority.pth')

    valuation_high_priority = pta.DqnAgent(
        observation_shape=observation_shape,
        number_of_actions=4,
        gamma=0.95,
        epsilon_decay=0.00002,
        epsilon_min=0.0,
        mini_batch_size=64,
        warm_up_duration=1000,
        buffer_capacity=20000,
        target_update_period=2000,
        seed=1337)
    valuation_high_priority.epsilon = 0.01
    valuation_high_priority.load_weights('valuation_nets/high_priority.pth')

    valuation_nets = [valuation_low_priority, valuation_high_priority]

    trade = Trade(valuation_nets=valuation_nets,
                  agents=agents,
                  trading=params.trading,
                  n_trade_steps=params.trading_steps,
                  mark_up=params.mark_up,
                  gamma=params.gamma,
                  pay_up_front=params.pay_up_front,
                  trading_budget=params.trading_budget)

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
        q_vals = [[], []]
        accumulated_transfer = np.zeros(len(agents))
        combined_frames = drawing_util.render_combined_frames(combined_frames, env, [0, 0], 0, [0, 0])


        while not done:

            actions = []
            for agent_index in agent_indices:
                if not joint_done[agent_index]:
                    if params.trading == 2:
                        tr_checks = trade.check_actions(trade.suggested_steps)
                        if tr_checks[agent_index]:
                            action = agents[agent_index].policy(observations[agent_index]) + 4
                        else:
                            action = no_tr_agents[agent_index].policy(observations[agent_index])
                    else:
                        action = agents[agent_index].policy(observations[agent_index])
                else:
                    action = np.random.randint(0, 4)
                actions.append(action)

            joint_reward, observations, joint_done, info = trade.trading_step(episode_return, env, actions)

            # finish current step
            current_step += 1

            for i in range(trade.agent_count):
                episode_return[i] += joint_reward[i]
                if trade.n_trade_steps > 0 and trade.trading > 0:
                    trade_count[i] += info['new_trades_{}'.format(i)]
                    accumulated_transfer[i] += info['act_transfer_{}'.format(i)]

                q_vals[i] = agents[i].compute_q_values(observations[i])

            if not done:
                info_trade = 0
                if trade.n_trade_steps > 0 and trade.trading > 0:
                    for i in range(trade.agent_count):
                        if info['new_trades_{}'.format(i)] != 0:
                            info_trade = 1
                for i in range(3):
                    combined_frames = drawing_util.render_combined_frames(combined_frames, env, episode_return, info_trade, actions, [0, 0])

            done = all(done is True for done in joint_done) or current_step == episode_steps

        print("rewards: ", episode_return)
        print("trades: ", trade_count)
        print("transfer: ", accumulated_transfer)

    export_video('Smart-Factory-Trading.mp4', combined_frames, None)


if __name__ == '__main__':
    main()
