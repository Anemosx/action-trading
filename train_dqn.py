import os
from dotmap import DotMap
from agent import build_agent, fit_n_agents_n_step_contracting, test_n_agents_n_step_contracting
from envs.smartfactory import Smartfactory
from common_utils.utils import save_params
from visualization import TensorBoardLogger
from datetime import datetime
from contracting import Contract
import json


def train():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    log_dir = os.path.join(os.getcwd(), 'experiments', '{}'.format(datetime.now().strftime('%Y%m%d-%H-%M-%S')))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for run in range(1):
        for c in [0]:

            params.contracting = c
            run_dir = os.path.join(log_dir, 'run-{}'.format(run), 'contracting-{}'.format(c))
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            tensorboard_logger = TensorBoardLogger(log_dir=run_dir)
            tensorboard_logger.compile()
            model_dir = os.path.join(run_dir, 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            env = Smartfactory(nb_agents=params.nb_agents,
                               field_width=params.field_width,
                               field_height=params.field_height,
                               rewards=params.rewards,
                               contracting=c,
                               nb_machine_types=params.nb_machine_types,
                               nb_tasks=params.nb_tasks
                               )

            processor = env.SmartfactoryProcessor()
            params.nb_actions = env.nb_actions
            save_params(params, run_dir)

            agents = []
            for _ in range(params.nb_agents):
                agent = build_agent(params=params, processor=processor)
                agents.append(agent)
            # agents[0].load_weights('experiments/20190917-18-08-13/run-0/contracting-0/dqn_weights-agent-0.h5f')
            # agents[1].load_weights('experiments/20190917-18-08-13/run-0/contracting-0/dqn_weights-agent-1.h5f')

            contract = None
            if c:
                assert params.nb_agents == 2
                params.nb_actions = params.nb_actions_no_contracting_action
                contracting_agents = []
                for i in range(params.nb_agents):
                    agent = build_agent(params=params, processor=processor)
                    # agent.load_weights(os.path.join(log_dir, 'run-{}'.format(run), 'contracting-0/dqn_weights-agent-{}.h5f'.format(i)))
                    agent.load_weights('experiments/20190918-10-04-56/run-0/contracting-0/dqn_weights-agent-{}.h5f'.format(i))
                    contracting_agents.append(agent)

                contract = Contract(agent_1= contracting_agents[0],  # agents[0],
                                    agent_2= contracting_agents[1],  # agents[1],
                                    contracting_actions=c,
                                    contracting_target_update=params.contracting_target_update,
                                    nb_contracting_steps=params.nb_contracting_steps,
                                    mark_up=params.mark_up)

                if c == 1:
                    params.nb_actions = params.nb_actions_one_contracting_action
                if c == 2:
                    params.nb_actions = params.nb_actions_two_contracting_action

            fit_n_agents_n_step_contracting(env,
                                            agents=agents,
                                            nb_steps=params.nb_steps,
                                            nb_max_episode_steps=90,
                                            logger=tensorboard_logger,
                                            log_dir=run_dir,
                                            contract=contract)

            for i_agent, agent in enumerate(agents):
                agent.save_weights(os.path.join(run_dir, 'dqn_weights-agent-{}.h5f'.format(i_agent)), overwrite=True)

            test_n_agents_n_step_contracting(env,
                                             agents=agents,
                                             nb_episodes=20,
                                             nb_max_episode_steps=50,
                                             log_dir=run_dir,
                                             contract=contract)


if __name__ == '__main__':
    train()
