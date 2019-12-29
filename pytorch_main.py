import os

import agents.pytorch_agents as pta
from agents.pytorch_agents import make_dqn_agent
import pytorch_training
import pytorch_evaluation
from envs.smartfactory import Smartfactory, make_smart_factory
from dotmap import DotMap
import json
import neptune
from datetime import datetime
from contracting import Contract
import trading


def main():

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    # eval_mode:     0: eval_trading_steps
    #                1: eval_trading_budget
    #                2: eval_mark_up
    #               -1: train valuation nets

    # trading_mode:  0: exploding action space
    #                1: split between action and suggestion agents
    #                2: suggestion with extra observation channel

    mode_str, eval_list = trading.eval_mode_setup(params)

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    log_dir = os.path.join(os.getcwd(), 'exp-trading', '{} - tr mode {} - {}'.format(mode_str, params.trading_mode, exp_time))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for i_values in eval_list:
        if params.eval_mode == 0:
            params.trading_steps = i_values
        if params.eval_mode == 1:
            params.trading_budget[0] = i_values
            params.trading_budget[1] = i_values
        if params.eval_mode == 2:
            params.mark_up = i_values

        log_dir_i = os.path.join(log_dir, '{} {}'.format(mode_str, i_values))
        if not os.path.exists(log_dir_i):
            os.makedirs(log_dir_i)

        if params.logging:

            neptune.init('arno/trading-agents',
                         api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIzMDc2ZmU2YS1lYWFkLTQwNjUtOTgyMS00OTczMGU4NDYzNzcifQ==')

            # neptune.init('Trading-Agents/Trading-Agents',
            #              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIzMDc2ZmU2YS1lYWFkLTQwNjUtOTgyMS00OTczMGU4NDYzNzcifQ==')

            logger = neptune
            with neptune.create_experiment(name='contracting-agents',
                                           params=params_json):

                neptune.append_tag('trading-steps-{}'.format(params.trading_steps))
                neptune.append_tag('trading-budget-{}'.format(params.trading_budget[0]))
                neptune.append_tag('mark-up-{}'.format(params.mark_up))
                neptune.append_tag('trading-mode-{}'.format(params.trading_mode))
                run_trade_experiment(params, logger, log_dir_i)
        else:
            logger = None
            run_trade_experiment(params, logger, log_dir_i)


def run_experiment(params, logger):

    TRAIN = True
    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       priorities=params.priorities,
                       contracting=params.contracting,
                       nb_machine_types=params.nb_machine_types,
                       nb_steps_machine_inactive=params.nb_steps_machine_inactive,
                       nb_tasks=params.nb_tasks,
                       observation=1
                       )

    observation_shape = list(env.observation_space.shape)
    number_of_actions = env.action_space.n

    agents = []
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

    policy_net = None
    if params.contracting > 0:
        policy_net = pta.DqnAgent(
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
        policy_net.epsilon = 0.01
        policy_net.load_weights('/Users/kyrill/Documents/research/contracting-agents/weights.0.pth')

    contract = Contract(policy_net=policy_net,
                        valuation_nets=[policy_net, policy_net],
                        contracting_target_update=params.contracting_target_update,
                        gamma=params.gamma,
                        nb_contracting_steps=params.nb_contracting_steps,
                        mark_up=params.mark_up,
                        render=False)

    if TRAIN:
        pytorch_training.train_dqn(agents, env, 1000, params.nb_max_episode_steps, "id", logger, True, contract)

        for i_agent, agent in enumerate(agents):
            ag.save_weights("weights.{}.pth".format(i_agent))
    else:
        for i_agent, agent in enumerate(agents):
            agent.load_weights("weights.{}.pth".format(i_agent))
            agent.epsilon = agent.epsilon_min  # only for dqn_agent
        pytorch_evaluation.evaluate(agents, env, 100, 100, 'id', False, True, False, logger)


def run_trade_experiment(params, logger, log_dir):

    env = make_smart_factory(params)
    observation_shape = list(env.observation_space.shape)
    number_of_actions = env.action_space.n
    if params.eval_mode < 0:
        env.set_valuation_training(True)

    agents = []
    suggestion_agents = []
    for i_ag in range(params.nb_agents):
        ag = make_dqn_agent(params, observation_shape, number_of_actions)
        agents.append(ag)
    if params.trading_mode == 1:
        for i_ag in range(params.nb_agents):
            suggestion_ag = make_dqn_agent(params, observation_shape, number_of_actions)
            suggestion_agents.append(suggestion_ag)
    if params.trading_mode == 2:
        suggestion_agents = agents

    trade = trading.Trade(env=env, params=params, agents=agents, suggestion_agents=suggestion_agents)

    pytorch_training.train_trading_dqn(agents, env, 1500, params.nb_max_episode_steps, logger, trade, params.trading_mode, params.trading_budget)

    if params.eval_mode >= 0:
        for i_agent in range(len(agents)):
            agents[i_agent].save_weights(os.path.join(log_dir, 'weights-{}.pth'.format(i_agent)))
        if params.trading_mode == 1:
            for i_sugg_agent in range(len(suggestion_agents)):
                suggestion_agents[i_sugg_agent].save_weights(os.path.join(log_dir, 'weights-sugg-{}.pth'.format(i_sugg_agent)))
    else:
        agents[0].save_weights(os.path.join('high_priority.pth'))
        agents[1].save_weights(os.path.join('low_priority.pth'))


if __name__ == '__main__':
    main()