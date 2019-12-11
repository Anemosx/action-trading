import agents.pytorch_agents as pta
import pytorch_training
import pytorch_evaluation
from envs.smartfactory import Smartfactory
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

    if params.logging:
        # neptune.init('kyrillschmid/contracting-agents',
        #              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIzNTQ1ZWQwYy0zNzZiLTRmMmMtYmY0Ny0zN2MxYWQ2NDcyYzEifQ==')

        neptune.init('arno/trading-agents',
                     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIzMDc2ZmU2YS1lYWFkLTQwNjUtOTgyMS00OTczMGU4NDYzNzcifQ==')

        # neptune.init('Trading-Agents/Trading-Agents',
        #              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIzMDc2ZmU2YS1lYWFkLTQwNjUtOTgyMS00OTczMGU4NDYzNzcifQ==')

        logger = neptune
        with neptune.create_experiment(name='contracting-agents',
                                       params=params_json):

            exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            neptune.append_tag('pytorch-{}-trading-{}-'.format(exp_time, params.trading))
            run_trade_experiment(params, logger)
    else:
        logger = None
        run_trade_experiment(params, logger)


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


def run_trade_experiment(params, logger):

    TRAIN = True

    if params.trading > 0 and params.trading_steps > 0:
        action_space = trading.setup_action_space(params.trading_steps, params.trading_steps, None)
    else:
        action_space = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]

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
                number_of_actions=number_of_actions-4,
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

    trade = trading.Trade(valuation_nets=valuation_nets,
                          agents=agents,
                          trading=params.trading,
                          n_trade_steps=params.trading_steps,
                          mark_up=params.mark_up,
                          gamma=params.gamma,
                          pay_up_front=params.pay_up_front,
                          trading_budget=params.trading_budget)

    if TRAIN:
        pytorch_training.train_trading_dqn(agents, no_tr_agents, env, 2000, params.nb_max_episode_steps, "id", logger, False, trade, params.trading_budget)
        for i_agent, agent in enumerate(agents):
            ag.save_weights("weights.{}.pth".format(i_agent))
    else:
        for i_agent, agent in enumerate(agents):
            agent.load_weights("weights.{}.pth".format(i_agent))
            agent.epsilon = agent.epsilon_min  # only for dqn_agent
        pytorch_evaluation.evaluate(agents, env, 100, 100, 'id', False, True, False, logger)


if __name__ == '__main__':
    main()