import agents.pytorch_agents as pta
import pytorch_training
import pytorch_evaluation
from envs.smartfactory import Smartfactory
from dotmap import DotMap
import json
import neptune
from datetime import datetime
from contracting import Contract


def main():

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    if params.logging:
        neptune.init('kyrillschmid/contracting-agents',
                     api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIzNTQ1ZWQwYy0zNzZiLTRmMmMtYmY0Ny0zN2MxYWQ2NDcyYzEifQ==')
        logger = neptune
        with neptune.create_experiment(name='contracting-agents',
                                       params=params_json):

            exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
            neptune.append_tag('pytorch-{}-contracting-{}-'.format(exp_time, params.contracting))
            run_experiment(params, logger)
    else:
        logger = None
        run_experiment(params, logger)


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



if __name__ == '__main__':
    main()