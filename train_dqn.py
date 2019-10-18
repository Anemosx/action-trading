import os
from dotmap import DotMap
from agent import build_agent, fit_n_agents_n_step_contracting, test_n_agents_n_step_contracting
from envs.smartfactory import Smartfactory
from common_utils.utils import save_params
from visualization import TensorBoardLogger
from datetime import datetime
from contracting import Contract
import json
import neptune
import argparse


def train(setting, step_penalty):

    assert setting in [0, 2]
    assert step_penalty in [0, 1, 2]

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)
    params.contracting = setting

    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    step_penalties = [[0.4, 0.04]][step_penalty]

    log_dir = os.path.join(os.getcwd(), 'experiments', '{}'.format(exp_time), 'step-penalty-{}'.format(step_penalties[1]))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    neptune.init('kyrillschmid/contracting-agents',
                 api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5tbCIsImFwaV9rZXkiOiIzNTQ1ZWQwYy0zNzZiLTRmMmMtYmY0Ny0zN2MxYWQ2NDcyYzEifQ==')



    with neptune.create_experiment(name='contracting-agents',
                                   params=params_json):

        neptune.append_tag('{}-contracting-{}-step_penalty-{}'.format(exp_time, setting, step_penalties[1]))

        run_dir = os.path.join(log_dir, 'run-{}'.format(step_penalties[1]), 'contracting-{}'.format(setting))
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # tensorboard_logger = TensorBoardLogger(log_dir=run_dir)
        # tensorboard_logger.compile()
        model_dir = os.path.join(run_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        env = Smartfactory(nb_agents=params.nb_agents,
                           field_width=params.field_width,
                           field_height=params.field_height,
                           rewards=params.rewards,
                           step_penalties=step_penalties,
                           contracting=setting,
                           nb_machine_types=params.nb_machine_types,
                           nb_tasks=params.nb_tasks
                           )

        processor = env.SmartfactoryProcessor()
        params.nb_actions = env.nb_actions
        params.step_penalties = step_penalties
        save_params(params, run_dir)

        agents = []
        for _ in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=env.nb_actions,  processor=processor)
            agents.append(agent)
        # agents[0].load_weights('experiments/20190923-10-58-52/run-0/contracting-2/dqn_weights-agent-0.h5f')
        # agents[1].load_weights('experiments/20190923-10-58-52/run-0/contracting-2/dqn_weights-agent-1.h5f')

        contract = None
        if setting > 0:
            contracting_agents = []
            for i in range(params.nb_agents):
                agent = build_agent(params=params, nb_actions=params.nb_actions_no_contracting_action,
                                    processor=processor)
                agent.load_weights('experiments/20191014-17-56-01/run-0/contracting-0/dqn_weights-agent-{}.h5f'.format(0))
                contracting_agents.append(agent)

            contract = Contract(agent_1=contracting_agents[0],
                                agent_2=contracting_agents[1],
                                contracting_target_update=params.contracting_target_update,
                                nb_contracting_steps=params.nb_contracting_steps,
                                mark_up=params.mark_up)

        fit_n_agents_n_step_contracting(env,
                                        agents=agents,
                                        nb_steps=params.nb_steps,
                                        nb_max_episode_steps=120,
                                        logger=neptune,
                                        log_dir=run_dir,
                                        contract=contract)

        for i_agent, agent in enumerate(agents):
            agent.save_weights(os.path.join(run_dir, 'dqn_weights-agent-{}.h5f'.format(i_agent)), overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("setting", help="Choose the setting, i.e., 0-number_settings", metavar="SETTING", type=int)
    parser.add_argument("step_penalty", help="Choose the step_penalty", metavar="STEP_PENALTY", type=int)
    args = parser.parse_args()
    train(args.setting, args.step_penalty)
