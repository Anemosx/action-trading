import os
from dotmap import DotMap
from agent import build_agent, fit_n_agents, test_n_agents, fit_n_agents_n_step_contracting, test_n_agents_n_step_contracting
from envs.MAWicksellianTriangle import MAWicksellianTriangle, decentral_learning
from common_utils.utils import save_params
from visualization import TensorBoardLogger
from datetime import datetime
from contracting import Contract
import json


def train():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    log_dir = os.path.join(os.getcwd(), 'logs', '{}'.format(datetime.now().strftime('%Y%m%d-%H-%M-%S')))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for c in [False]:
        params.contracting = c
        run_dir = os.path.join(log_dir, 'contracting-{}'.format(c))
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        tensorboard_logger = TensorBoardLogger(log_dir=run_dir)
        tensorboard_logger.compile()
        model_dir = os.path.join(run_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        env = MAWicksellianTriangle(nb_agents=2,
                                    field_width=params.field_width,
                                    field_height=params.field_height,
                                    rewards=params.rewards,
                                    learning=decentral_learning,
                                    contracting=c)

        processor = env.MAWicksellianTriangleProcessor()
        params.nb_actions = env.nb_actions
        save_params(params, log_dir)

        agents = []
        for _ in range(params.nb_learners):
            agent = build_agent(params=params, processor=processor)
            agents.append(agent)
        # agents[0].load_weights('logs/20190907-20-27-10/contracting-True/dqn_weights-agent-0.h5f')
        # agents[1].load_weights('logs/20190907-20-27-10/contracting-True/dqn_weights-agent-1.h5f')

        contract = None
        if c:
            params.nb_actions = 4
            contracting_agents = []
            for i in range(2):
                agent = build_agent(params=params, processor=processor)
                # agent.load_weights(os.path.join(log_dir, 'contracting-False/dqn_weights-agent-{}.h5f'.format(i)))
                contracting_agents.append(agent)
            contracting_agents[0].load_weights('logs/20190906-10-20-15/contracting-False/dqn_weights-agent-0.h5f')
            contracting_agents[1].load_weights('logs/20190906-10-20-15/contracting-False/dqn_weights-agent-1.h5f')
            contract = Contract(agent_1=contracting_agents[0], agent_2=contracting_agents[1])
            params.nb_actions = 12

        fit_n_agents_n_step_contracting(env,
                                        agents=agents,
                                        nb_steps=params.nb_steps,
                                        nb_max_episode_steps=200,
                                        logger=tensorboard_logger,
                                        contract=contract)

        for i_agent, agent in enumerate(agents):
            agent.save_weights(os.path.join(run_dir, 'dqn_weights-agent-{}.h5f'.format(i_agent)), overwrite=True)

        test_n_agents_n_step_contracting(env,
                                         agents=agents,
                                         nb_episodes=25,
                                         nb_max_episode_steps=45,
                                         log_dir=run_dir,
                                         contract=contract)


if __name__ == '__main__':
    train()
