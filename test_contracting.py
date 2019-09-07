import os
from dotmap import DotMap
from agent import build_agent, fit_n_agents, test_n_agents
from envs.MAWicksellianTriangle import MAWicksellianTriangle, decentral_learning

from contracting import Contract
import json

import numpy as np
from copy import deepcopy
from collections import deque
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def test(log_dir):

    i_episodes = 0
    df = pd.DataFrame(columns=('episode',
                               'contracting',
                               'reward',
                               'reward_a1', 'reward_a2',
                               'episode_compensations_a1', 'episode_compensations_a2',
                               'number_contracts',
                               'episode_steps'))
    nb_max_episode_steps = 100
    nb_episodes = 100

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    for c in [False, True]:
        params.contracting = c

        env = MAWicksellianTriangle(nb_agents=2,
                                    field_width=params.field_width,
                                    field_height=params.field_height,
                                    rewards=params.rewards,
                                    learning=decentral_learning,
                                    contracting=c)

        processor = env.MAWicksellianTriangleProcessor()
        params.nb_actions = env.nb_actions

        agents = []
        for _ in range(params.nb_learners):
            agent = build_agent(params=params, processor=processor)
            agents.append(agent)
        if c:
            agents[0].load_weights(os.path.join(log_dir, 'contracting-True/dqn_weights-agent-0.h5f'))
            agents[1].load_weights(os.path.join(log_dir, 'contracting-True/dqn_weights-agent-1.h5f'))
        else:
            agents[0].load_weights(os.path.join(log_dir, 'contracting-False/dqn_weights-agent-0.h5f'))
            agents[1].load_weights(os.path.join(log_dir, 'contracting-False/dqn_weights-agent-1.h5f'))

        contract = None
        if c:
            params.nb_actions = 4
            contracting_agents = []
            for i in range(2):
                agent = build_agent(params=params, processor=processor)
                agent.training = False
                contracting_agents.append(agent)
            contracting_agents[0].load_weights(os.path.join(log_dir, 'contracting-False/dqn_weights-agent-0.h5f'))
            contracting_agents[1].load_weights(os.path.join(log_dir, 'contracting-False/dqn_weights-agent-1.h5f'))
            contract = Contract(agent_1=contracting_agents[0], agent_2=contracting_agents[1])
            params.nb_actions = 12

        for i, agent in enumerate(agents):
            agent.training = False

        for episode in range(nb_episodes):
            episode_rewards = [0. for _ in agents]
            episode_step = 0
            nb_contracts = 0
            plan = []
            episode_compensations = np.zeros(2)
            observations = deepcopy(env.reset())

            for i, agent in enumerate(agents):
                episode_rewards[i] = 0.
                # Obtain the initial observation by resetting the environment.
                agent.reset_states()
                if agent.processor is not None:
                    observations[i] = agent.processor.process_observation(observations[i])
                assert observations[i] is not None

            # Run the episode until we're done.
            done = False
            while not done:
                actions = []
                if len(plan) > 0:
                    plan_action = plan.popleft()
                    actions.append(plan_action[0])
                    actions.append(plan_action[1])
                else:
                    for i, agent in enumerate(agents):
                        actions.append(agent.forward(observations[i]))
                        if agent.processor is not None:
                            actions[i] = agent.processor.process_action(actions[i])

                if contract is not None:
                    if len(plan) == 0:
                        plan, compensations = contract.find_contract(env=env,
                                                                     actions=actions,
                                                                     observations=observations)
                        plan = deque(plan)
                        if len(plan) > 0:
                            nb_contracts += 1
                        episode_compensations += compensations

                observations, r, d, info = env.step(actions)
                observations = deepcopy(observations)

                if contract is not None:
                    r, episode_compensations = contract.get_compensated_rewards(agents=agents,
                                                                                rewards=r,
                                                                                episode_compensations=episode_compensations)

                for i, agent in enumerate(agents):
                    if agent.processor is not None:
                        observations[i], r[i], done, info = agent.processor.process_step(observations[i], r[i], d, info)

                if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                    done = True

                for i, agent in enumerate(agents):
                    episode_rewards[i] += r[i]
                episode_step += 1
                i_episodes += 1

            df.loc[i_episodes] = [episode,
                                  c,
                                  episode_rewards[0] + episode_rewards[1],
                                  episode_rewards[0], episode_rewards[1],
                                  episode_compensations[0], episode_compensations[1],
                                  nb_contracts,
                                  episode_step]
    return df


if __name__ == '__main__':
    log_dir = 'logs/20190907-10-58-39'
    df = test(log_dir)
    df.to_csv(os.path.join(log_dir,  'test-values.csv'))
    df = pd.read_csv(os.path.join(log_dir,  'test-values.csv'))

    sns.boxplot(x="contracting", y="reward",
                data=df, palette="Set3")
    plt.savefig(os.path.join(log_dir, 'rewards-var.png'))
    plt.clf()

    sns.boxplot(x="contracting", y="reward_a1",
                data=df, palette="Set3")
    plt.savefig(os.path.join(log_dir, 'rewards_a1-var.png'))
    plt.clf()

    sns.boxplot(x="contracting", y="reward_a2",
                data=df, palette="Set3")
    plt.savefig(os.path.join(log_dir, 'rewards_a2-var.png'))
    plt.clf()
