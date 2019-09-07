import numpy as np
from copy import deepcopy
from envs.MAWicksellianTriangle import MAWicksellianTriangle
from common_utils.utils import export_video
from agent import build_agent
from collections import deque
from dotmap import DotMap
import json
import scipy

ONE_CONTRACTING_ACTION = 'ONE_CONTRACTING_ACTION'
TWO_CONTRACTING_ACTIONS = 'TWO_CONTRACTING_ACTIONS'


class Contract:

    def __init__(self, agent_1, agent_2):
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.agents = [agent_1, agent_2]

    def find_contract(self, env, actions, observations, modus=TWO_CONTRACTING_ACTIONS):
        """
        This method decides on whether contracting takes places and returns a list
        of actions that should be executed and the compensation for agents under contracting

        :param env: environment in its current state
        :param actions: actions from current state and policy
        :param observations: current observations from agents
        :param modus: the contracting modus
        :return: list of actions to be executed
        """

        nb_plans = 10
        nb_actions = 10
        env_nb_actions = len(self.agents[0].compute_q_values(observations[0]))

        # test for contracting conditions
        a1_greedy, a2_greedy = 0, 0

        if env.actions[actions[0]][2] == 1 and env.actions[actions[1]][3]:
            a2_greedy = 1
        if env.actions[actions[0]][3] == 1 and env.actions[actions[1]][2]:
            a1_greedy = 1
        if not a1_greedy and not a2_greedy:
            return [], [0, 0]

        assert any(i == 0 for i in [a1_greedy, a2_greedy])

        # 1. Find contracting plan to be executed
        state = env.get_state()

        actions_a1 = np.random.randint(0, env_nb_actions, (nb_plans, nb_actions, 2), np.int64)
        actions_a2 = np.random.randint(0, env_nb_actions, (nb_plans, nb_actions, 2), np.int64)
        actions = [actions_a1, actions_a2]

        greedy_q_vals_a1 = np.zeros(nb_plans)
        greedy_q_vals_a2 = np.zeros(nb_plans)
        greedy_q_vals = [greedy_q_vals_a1, greedy_q_vals_a2]

        non_greedy_q_vals_a1 = np.zeros(nb_plans)
        non_greedy_q_vals_a2 = np.zeros(nb_plans)
        non_greedy_q_vals = [non_greedy_q_vals_a1, non_greedy_q_vals_a2]

        for greedy_agent, non_greedy_agent in [[0, 1], [1, 0]]:

            for plan in range(nb_plans):
                env.set_state(state)
                last_observations = observations
                for action in range(nb_actions):
                    ac = actions[greedy_agent][plan, action]

                    observations[greedy_agent] = self.agents[greedy_agent].processor.process_observation(
                        observations[greedy_agent])

                    ac[greedy_agent] = self.agents[greedy_agent].forward(observations[greedy_agent])
                    actions[greedy_agent][plan, action] = ac

                    observations, r, done, info = env.step(actions[greedy_agent][plan, action])
                    observations = deepcopy(observations)

                    # TODO: Action values from observations with done=True
                    # are not good estimated -> should use last observation

                    if done:
                        observations = last_observations
                        break

                    last_observations = observations

                for i, agent in enumerate(self.agents):
                    if agent.processor is not None:
                        observations[i] = agent.processor.process_observation(observations[i])
                greedy_q_vals[greedy_agent][plan] = np.max(
                    self.agents[greedy_agent].compute_q_values(observations[greedy_agent]))

                non_greedy_q_vals[non_greedy_agent][plan] = np.max(
                    self.agents[non_greedy_agent].compute_q_values(observations[non_greedy_agent]))

        env.set_state(state)

        # 3. Calculate compensation
        """
        1. Agents can make debts in the form of value that will be compensated later. 
        An agent pays back the loss of value in terms of reward when 
        the reward is actually experienced. As the actual payment is delayed it also needs to be
        discounted (what about step penalty?).
        
        2. Should an agent receive a markup to be overcompensated?
        
        3. When does an agent receive its compensation?
        
        4. Explicit signalling: agents have to signal contracting in an earlier step.
        
        5. Max contracting depth = 1?
        """

        best_opt_a1 = np.argmax(greedy_q_vals_a1)
        max_q_val_a1 = np.max(greedy_q_vals_a1)

        best_opt_a2 = np.argmax(greedy_q_vals_a2)
        max_q_val_a2 = np.max(greedy_q_vals_a2)

        delta = 0.4
        compensation_val_a1 = 0
        compensation_val_a2 = 0

        if a1_greedy:
            diff_greedy_non_greedy_a2 = max(0, max_q_val_a2 - non_greedy_q_vals_a2[best_opt_a1])
            markup = delta * diff_greedy_non_greedy_a2
            compensation_val_a1 = diff_greedy_non_greedy_a2 + markup
            plan = actions_a1[best_opt_a1]
        elif a2_greedy:
            diff_greedy_non_greedy_a1 = max(0, max_q_val_a1 - non_greedy_q_vals_a1[best_opt_a2])
            markup = delta * diff_greedy_non_greedy_a1
            compensation_val_a2 = diff_greedy_non_greedy_a1 + markup
            plan = actions_a2[best_opt_a2]

        compensations = [compensation_val_a1, compensation_val_a2]

        return list(plan), compensations

    def check_contracting(self, env, actions):

        greedy = [False, False]
        contracting = False
        if env.actions[actions[0]][2] == 1 and env.actions[actions[1]][3] == 1:
            contracting = True
            greedy[1] = True
        if env.actions[actions[0]][3] == 1 and env.actions[actions[1]][2] == 1:
            contracting = True
            greedy[0] = True

        return contracting, greedy

    def contracting_n_steps(self, env, observations, greedy, frames=None, info_values=None):

        nb_contracting_steps = 10
        mark_up = 2

        compensations = np.zeros(2)
        for i_agent, agent in enumerate(self.agents):
            if not greedy[i_agent]:

                q_vals = self.agents[i_agent].compute_q_values(observations[i_agent])
                compensations[(i_agent + 1) % 2] = np.max(q_vals) * mark_up

        for c_step in range(nb_contracting_steps):
            c_actions = []
            for i_agent, agent in enumerate(self.agents):
                if greedy[i_agent]:
                    c_actions.append(self.agents[i_agent].forward(observations[i_agent]))
                else:
                    q_vals = self.agents[i_agent].compute_q_values(observations[i_agent])
                    c_actions.append(np.argmin(q_vals))

            observations, r, done, info = env.step(c_actions)
            observations = deepcopy(observations)

            if frames is not None:
                if info_values is not None:
                    for i, agent in enumerate(env.agents):
                        info_values[i]['reward'] = r[i]
                        info_values[i]['agent_comp'] = compensations[i]
                        info_values[i]['contracting'] = 1
                        info_values[i]['a{}_greedy'.format(i)] = greedy[i]

                frames.append(env.render(mode='rgb_array', info_values=info_values))

            if done:
                break

        return observations, r, done, info, compensations

    @staticmethod
    def get_compensated_rewards(agents, rewards, episode_compensations):

        if np.count_nonzero(episode_compensations) == 0 or (rewards <= 0).all():
            return rewards, episode_compensations
        else:
            # clear compensation with each other
            if episode_compensations[0] > 0 and episode_compensations[1] > 0:
                diff_a1 = np.maximum(episode_compensations[0] - episode_compensations[1], 0)
                diff_a2 = np.maximum(episode_compensations[1] - episode_compensations[0], 0)
                episode_compensations = [diff_a1, diff_a2]

            r = [0, 0]
            transfer = [0, 0]
            for i, agent in enumerate(agents):
                if episode_compensations[i] > 0:
                    if rewards[i] <= 0:
                        transfer[i] = 0
                    elif episode_compensations[i] >= rewards[i]:
                        transfer[i] = rewards[i]
                    else:
                        transfer[i] = episode_compensations[i]
                    episode_compensations[i] -= transfer[i]
                    episode_compensations[i] = np.maximum(episode_compensations[i], 0)

            r[0] += rewards[0] - transfer[0] + transfer[1]
            r[1] += rewards[1] - transfer[1] + transfer[0]
        return r, episode_compensations


def main():
    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    env = MAWicksellianTriangle(nb_agents=params.nb_learners,
                                field_width=params.field_width,
                                field_height=params.field_height,
                                rewards=params.rewards,
                                contracting=True)
    processor = env.MAWicksellianTriangleProcessor()

    params.nb_actions = env.nb_actions
    episodes = 10
    episode_steps = 100

    contracting_agents = []
    params.nb_actions = 4
    for i in range(2):
        agent = build_agent(params=params, processor=processor)
        contracting_agents.append(agent)
    contracting_agents[0].load_weights('logs/20190905-15-47-42/contracting-False/dqn_weights-agent-0.h5f')
    contracting_agents[1].load_weights('logs/20190905-15-47-42/contracting-False/dqn_weights-agent-1.h5f')
    contract = Contract(agent_1=contracting_agents[0], agent_2=contracting_agents[1])
    params.nb_actions = 12

    agents = []
    for _ in range(params.nb_learners):
        agent = build_agent(params=params, processor=processor)
        agents.append(agent)

    frames = []
    for i_episode in range(episodes):

        observations = env.reset()
        episode_compensations = np.zeros(2)
        plan = []
        info_values = [{'reward': 0.0,
                        'agent_comp': episode_compensations[i],
                        'contracting': len(plan),
                        'a{}_greedy'.format(i): -1,
                        } for i in range(env.nb_agents)]
        frames.append(env.render(mode='rgb_array', info_values=info_values))

        for i_step in range(episode_steps):

            actions = []
            #for i, agent in enumerate(agents):
            #    observations[i] = agent.processor.process_observation(observations[i])
            #    actions.append(agent.forward(observations[i]))
            #    if agent.processor is not None:
            #        actions[i] = agent.processor.process_action(actions[i])
            actions = [np.random.randint(0, env.nb_actions, 1), np.random.randint(0, env.nb_actions, 1)]

            contracting = False
            if contract is not None:
                contracting, greedy = contract.check_contracting(env, actions)

            accumulated_info = {}
            done = False

            env.contract = contracting
            if contracting:
                observations, r, done, info, compensation = contract.contracting_n_steps(env,
                                                                                         observations,
                                                                                         greedy,
                                                                                         frames,
                                                                                         info_values)
                episode_compensations += compensation
            else:
                observations, r, done, info = env.step(actions)
                observations = deepcopy(observations)

            r, episode_compensations = contract.get_compensated_rewards(agents=agents,
                                                                        rewards=r,
                                                                        episode_compensations=episode_compensations)

            for i_ag in range(2):
                scipy.misc.toimage(observations[i_ag], cmin=0.0, cmax=...).save('observations/new-outfile-{}-ag-{}.jpg'.format(i_step, i_ag))

            for i, agent in enumerate(env.agents):
                info_values[i]['reward'] = r[i]
                info_values[i]['agent_comp'] = episode_compensations[i]
                info_values[i]['contracting'] = 0
                info_values[i]['a{}_greedy'.format(i)] = greedy[i]

            frames.append(env.render(mode='rgb_array', info_values=info_values))

            if done:
                break

    export_video('MAWicksellianTriangle-Contracting.mp4', frames, None)

if __name__ == '__main__':
    main()
