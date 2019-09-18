import os
from dotmap import DotMap
from agent import build_agent, fit_n_self_play, test_n_agents_n_step_contracting
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

        params.rewards = [1, 1]

        run_dir = os.path.join(log_dir, 'run-{}'.format(run), 'self-play')
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        tensorboard_logger = TensorBoardLogger(log_dir=run_dir)
        tensorboard_logger.compile()
        model_dir = os.path.join(run_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        env = Smartfactory(nb_agents=2,
                           field_width=params.field_width,
                           field_height=params.field_height,
                           rewards=params.rewards,
                           contracting=False,
                           nb_machine_types=params.nb_machine_types,
                           nb_tasks=params.nb_tasks
                           )

        processor = env.SmartfactoryProcessor()
        params.nb_actions = env.nb_actions
        save_params(params, run_dir)

        agents = []
        for _ in range(params.nb_learners):
            agent = build_agent(params=params, processor=processor)
            agents.append(agent)
        agents[0].load_weights('experiments/20190918-12-59-40/run-0/self-play/dqn_weights-agent-1.h5f')
        agents[1].load_weights('experiments/20190918-12-59-40/run-0/self-play/dqn_weights-agent-1.h5f')

        fit_n_self_play(env,
                        agents=agents,
                        nb_steps=params.nb_steps,
                        nb_max_episode_steps=90,
                        logger=tensorboard_logger,
                        log_dir=run_dir,
                        contract=None,
                        self_play_update=100)

        for i_agent, agent in enumerate(agents):
            agent.save_weights(os.path.join(run_dir, 'dqn_weights-agent-{}.h5f'.format(i_agent)), overwrite=True)

if __name__ == '__main__':
    train()
