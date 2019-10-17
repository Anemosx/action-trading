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

class Contract:

    def __init__(self,
                 agent_1,
                 agent_2,
                 contracting_target_update=0,
                 nb_contracting_steps=10,
                 mark_up=1.0):

        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.agents = [agent_1, agent_2]
        self.contracting_target_update = contracting_target_update
        self.nb_contracting_steps = nb_contracting_steps
        self.mark_up = mark_up

    def contracting_n_steps(self, env, observations, actions, combined_frames=None, info_values=None):

        contracting, greedy = env.check_contracting(actions)

        if not contracting:
            observations, r, done, info = env.step(actions)
            return observations, r, done, info, contracting

        else:

            rewards = np.zeros(2)
            done = False
            info = None

            q_vals_a1 = self.agents[0].compute_q_values(observations[0])
            q_vals_a2 = self.agents[1].compute_q_values(observations[1])
            q_vals = [q_vals_a1, q_vals_a2]

            for i_agent, agent in enumerate(self.agents):
                if not greedy[i_agent]:
                    transfer = np.maximum(np.max(q_vals[i_agent]), 0) * self.mark_up
                    env.agents[(i_agent + 1) % 2].episode_debts += transfer

            for c_step in range(self.nb_contracting_steps):
                c_actions = []
                for i_agent, agent in enumerate(self.agents):
                    if greedy[i_agent]:
                        c_actions.append(self.agents[i_agent].forward(observations[i_agent]))
                    else:
                        c_actions.append(np.argmin(q_vals[i_agent]))

                observations, r, done, info = env.step(c_actions)
                observations = deepcopy(observations)

                r, transfer = self.get_compensated_rewards(env=env, rewards=r)
                rewards += r

                if combined_frames is not None:
                    if info_values is not None:
                        for i, agent in enumerate(env.agents):
                            info_values[i]['a{}-reward'.format(i)] = r[i]
                            info_values[i]['a{}-episode_debts'.format(i)] = env.agents[i].episode_debts
                            info_values[i]['contracting'] = 1
                            info_values[i]['a{}-greedy'.format(i)] = greedy[i]
                            info_values[i]['a{}-q_max'.format(i)] = np.max(q_vals[i])

                    combined_frames = drawing_util.render_combined_frames(combined_frames, env, info_values)

                if any([agent.done for agent in env.agents]):
                    break

            return observations, rewards, done, info, contracting

    def get_compensated_rewards(self, env, rewards):

        transfer = [0, 0]
        if env.agents[0].episode_debts == 0 and env.agents[1].episode_debts == 0:
            return rewards, transfer
        elif (rewards <= 0).all():
            return rewards, transfer
        else:
            # clear compensation with each other
            if env.agents[0].episode_debts > 0 and env.agents[1].episode_debts > 0:
                diff_a1 = np.maximum(env.agents[0].episode_debts - env.agents[1].episode_debts, 0)
                diff_a2 = np.maximum(env.agents[1].episode_debts - env.agents[0].episode_debts, 0)
                env.agents[0].episode_debts = diff_a1
                env.agents[1].episode_debts = diff_a2

            r = [0, 0]
            for i, agent in enumerate(env.agents):
                if env.agents[i].episode_debts > 0:
                    if rewards[i] <= 0:
                        transfer[i] = 0
                    elif env.agents[i].episode_debts >= rewards[i]:
                        transfer[i] = rewards[i]
                    elif env.agents[i].episode_debts < rewards[i]:
                        transfer[i] = env.agents[i].episode_debts
                    env.agents[i].episode_debts -= transfer[i]
                    env.agents[i].episode_debts = np.maximum(env.agents[i].episode_debts, 0)

            r[0] += rewards[0] - transfer[0] + transfer[1]
            r[1] += rewards[1] - transfer[1] + transfer[0]
        return r, transfer


def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    c = 0
    policy_random = False
    episodes = 5
    episode_steps = 100

    ep_columns = ['episode', 'contracting', 'reward', 'number_contracts', 'episode_steps']
    for i_ag in range(params.nb_agents):
        ag_columns = ['reward_a{}'.format(i_ag),
                      'accumulated_transfer_a{}'.format(i_ag)]
        ep_columns += ag_columns
    df = pd.DataFrame(columns=ep_columns)

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       contracting=c,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks
                       )

    processor = env.SmartfactoryProcessor()

    contract = None
    if c > 0:
        contracting_agents = []
        for i in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=params.nb_actions_no_contracting_action, processor=processor)
            agent.load_weights('experiments/20191015-09-39-50/run-0/contracting-0/dqn_weights-agent-{}.h5f'.format(i))
            contracting_agents.append(agent)
        contract = Contract(agent_1=contracting_agents[0],
                            agent_2=contracting_agents[1],
                            contracting_target_update=params.contracting_target_update,
                            nb_contracting_steps=params.nb_contracting_steps,
                            mark_up=params.mark_up)

    agents = []
    for i_agent in range(params.nb_agents):
        agent = build_agent(params=params, nb_actions=env.nb_contracting_actions, processor=processor)
        agents.append(agent)
        agents[i_agent].load_weights('experiments/20191015-16-37-33/run-0/contracting-{}/dqn_weights-agent-{}.h5f'.format(c, i_agent))

    combined_frames = []
    for i_episode in range(episodes):

        observations = env.reset()
        episode_rewards = np.zeros(params.nb_agents)
        accumulated_transfer = np.zeros(params.nb_agents)
        contracting = False

        if contract is not None:
            q_vals_a1 = contract.agents[0].compute_q_values(observations[0])
            q_vals_a2 = contract.agents[1].compute_q_values(observations[1])
        else:
            q_vals_a1 = agents[0].compute_q_values(observations[0])
            q_vals_a2 = agents[1].compute_q_values(observations[1])
        q_vals = [q_vals_a1, q_vals_a2]

        info_values = [{'a{}-reward'.format(i): 0.0,
                        'a{}-episode_debts'.format(i): 0.0,
                        'contracting': 0,
                        'a{}-greedy'.format(i): 0,
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

            if contract is not None:
                observations, r, done, info, contracting = contract.contracting_n_steps(env, observations, actions,
                                                                                        combined_frames, info_values)
            else:
                observations, r, done, info = env.step(actions)

            observations = deepcopy(observations)

            if contract is not None and not any([agent.done for agent in env.agents]):
                r, transfer = contract.get_compensated_rewards(env=env, rewards=r)
                accumulated_transfer += transfer
            episode_rewards += r

            if not contracting:
                if contract is not None:
                    q_vals_a1 = contract.agents[0].compute_q_values(observations[0])
                    q_vals_a2 = contract.agents[1].compute_q_values(observations[1])
                else:
                    q_vals_a1 = agents[0].compute_q_values(observations[0])
                    q_vals_a2 = agents[1].compute_q_values(observations[1])
                q_vals = [q_vals_a1, q_vals_a2]
                for i, agent in enumerate(env.agents):
                    info_values[i]['a{}-reward'.format(i)] = r[i]
                    info_values[i]['a{}-episode_debts'.format(i)] = env.agents[i].episode_debts
                    info_values[i]['contracting'] = 0
                    info_values[i]['a{}-greedy'.format(i)] = 0
                    info_values[i]['a{}-q_max'.format(i)] = np.max(q_vals[i])
                    info_values[i]['a{}-done'.format(i)] = env.agents[i].done

                combined_frames = drawing_util.render_combined_frames(combined_frames, env, info_values, observations)

            if done:
                ep_stats = [i_episode, (contract is not None), np.sum(episode_rewards), 0,
                            episode_steps]
                for i_ag in range(len(agents)):
                    ag_stats = [episode_rewards[i_ag], accumulated_transfer[i_ag]]
                    ep_stats += ag_stats
                df.loc[i_episode] = ep_stats
                break

    df.to_csv(os.path.join('test-values-contracting-c-{}.csv'.format(c)))
    export_video('Smart-Factory-Contracting.mp4', combined_frames, None)

if __name__ == '__main__':
    main()
