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
        self.agent_count = 2
        self.n_trade_steps = n_trade_steps
        self.mark_up = mark_up

    # trading part of step

    def update_trading(self, env, actions, suggested_steps, new_trade, transfer):
        for i_agents in range(self.agent_count):
            if suggested_steps[i_agents] is []:
                new_trade[i_agents] = True
            else:
                new_trade[i_agents] = False

        observations, r, done, info = env.step(actions)
        observations = deepcopy(observations)

        act_transfer = np.zeros(self.agent_count)
        current_actions = env.get_current_actions()
        print(current_actions)

        for i_agents in range(self.agent_count):
            agent_of_action = current_actions[i_agents][0]
            if suggested_steps[0] != [] and suggested_steps[1] != []:
                if suggested_steps[agent_of_action][0] == current_actions[i_agents][1][0] and \
                        suggested_steps[agent_of_action][1] == current_actions[i_agents][1][1]:
                    del suggested_steps[agent_of_action][0]
                    del suggested_steps[agent_of_action][1]
                    if suggested_steps[agent_of_action] is None:
                        r, transfer, act_agent_transfer = self.pay_reward(agent_of_action % 2, agent_of_action, r, transfer)
                        act_transfer[agent_of_action] += act_agent_transfer
                else:
                    suggested_steps[agent_of_action] = []

            if new_trade[i_agents]:
                copy_action_from = i_agents % 2
                del current_actions[copy_action_from][0]
                del current_actions[copy_action_from][1]
                if current_actions[copy_action_from][0] != 0.0 and current_actions[copy_action_from][1] != 0.0:
                    for i_trading_steps in range(len(current_actions[copy_action_from][1])):
                        suggested_steps[i_agents].append(current_actions[copy_action_from][1][i_trading_steps])

                    if copy_action_from == 0:
                        q_val = self.agent_1.compute_q_values(observations[0])
                    else:
                        q_val = self.agent_2.compute_q_values(observations[1])
                    transfer[i_agents] = np.max(q_val)
                    new_trade[i_agents] = False

        return observations, r, done, info, suggested_steps, new_trade, transfer, act_transfer

    def pay_reward(self, payer, receiver, rewards, transfer_value):
        new_rewards = []
        new_transfer = []
        act_transfer = 0

        if rewards[payer]-transfer_value[receiver] > 0:
            new_rewards[payer] = rewards[payer] - transfer_value[receiver]
            new_rewards[receiver] = rewards[receiver] + transfer_value[receiver]
            act_transfer = transfer_value[receiver]
        else:
            new_rewards = rewards

        new_transfer[payer] = transfer_value[payer]
        new_transfer[receiver] = 0

        return new_rewards, new_transfer, act_transfer


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

            suggested_steps = [[],[]]
            new_trade = []
            transfer = []
            for i in range(params.nb_agents):
                new_trade.append(False)
                transfer.append(0)
            if trade is not None:
                q_vals_a1 = trade.agents[0].compute_q_values(observations[0])
                q_vals_a2 = trade.agents[1].compute_q_values(observations[1])
            else:
                q_vals_a1 = agents[0].compute_q_values(observations[0])
                q_vals_a2 = agents[1].compute_q_values(observations[1])
            q_vals = [q_vals_a1, q_vals_a2]

            info_values = [{'a{}-reward'.format(i): 0.0,
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

                if trade is not None and not any([agent.done for agent in env.agents]):
                    observations, r, done, info, suggested_steps, new_trade, transfer, act_transfer = trade.update_trading(env, actions, suggested_steps, new_trade, transfer)
                    accumulated_transfer += act_transfer
                else:
                    observations, r, done, info = env.step(actions)

                observations = deepcopy(observations)
                episode_rewards += r

                if trade is not None:
                    q_vals_a1 = trade.agents[0].compute_q_values(observations[0])
                    q_vals_a2 = trade.agents[1].compute_q_values(observations[1])
                else:
                    q_vals_a1 = agents[0].compute_q_values(observations[0])
                    q_vals_a2 = agents[1].compute_q_values(observations[1])
                q_vals = [q_vals_a1, q_vals_a2]
                for i, agent in enumerate(env.agents):
                    info_values[i]['a{}-reward'.format(i)] = r[i]
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
