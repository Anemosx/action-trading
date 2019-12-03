import os
import numpy as np
from copy import deepcopy
from envs.smartfactory import Smartfactory
from common_utils.utils import export_video
from agent import build_agent
from dotmap import DotMap
import json
import common_utils.drawing_util as drawing_util
import agents.pytorch_agents as pta

class Contract:

    def __init__(self,
                 policy_net,
                 valuation_nets,
                 gamma,
                 contracting_target_update=0,
                 nb_contracting_steps=10,
                 mark_up=1.0,
                 render=True):

        self.policy_net = policy_net
        self.valuation_nets = valuation_nets
        self.gamma = gamma
        self.contracting_target_update = contracting_target_update
        self.nb_contracting_steps = nb_contracting_steps
        self.mark_up = mark_up
        self.render = render

    def contracting_n_steps(self, env, observations, actions, combined_frames=None):

        contracting, greedy = env.check_contracting(actions)

        if not contracting:

            observations, r, done, info = env.step(actions)

            info['contracting'] =  0
            return observations, r, done, info

        else:

            pos_rewards_a0 = np.zeros((self.nb_contracting_steps))
            pos_rewards_a1 = np.zeros((self.nb_contracting_steps))

            neg_rewards_a0 = np.zeros((self.nb_contracting_steps))
            neg_rewards_a1 = np.zeros((self.nb_contracting_steps))

            compensations = np.zeros(self.nb_contracting_steps)

            done = False
            info = None
            priorities = env.priorities

            for c_step in range(self.nb_contracting_steps):

                c_actions = []
                for i_agent in range(2):
                    if greedy[i_agent]:
                        c_actions.append(self.policy_net.policy(observations[i_agent]))
                    else:
                        q_vals = self.valuation_nets[priorities[i_agent]].compute_q_values(observations[i_agent])
                        c_actions.append(np.argmin(q_vals))
                        c_t = np.maximum(np.max(q_vals) - np.min(q_vals), 0)
                        compensations[c_step] = c_t

                observations, r, done, info = env.step(c_actions)

                if r[0] >= 0:
                    pos_rewards_a0[c_step] += r[0]
                else:
                    neg_rewards_a0[c_step] += r[0]
                if r[1] >= 0:
                    pos_rewards_a1[c_step] += r[1]
                else:
                    neg_rewards_a1[c_step] += r[1]

                if any([agent.done for agent in env.agents]):
                    break

                compensations *= 1 / self.gamma

                if self.render and combined_frames is not None and c_step < self.nb_contracting_steps - 1:
                    combined_frames = drawing_util.render_combined_frames(combined_frames, env, r, 1, actions)

            r = [0, 0]
            accumulated_compensation = np.sum(compensations)
            if greedy[0]:

                transfer = np.minimum(np.sum(pos_rewards_a0), accumulated_compensation )
                r[0] += np.sum(pos_rewards_a0) + np.sum(neg_rewards_a0) - transfer
                r[1] += np.sum(pos_rewards_a1) + np.sum(neg_rewards_a1) + transfer
            elif greedy[1]:
                transfer = np.minimum(np.sum(pos_rewards_a1), accumulated_compensation)
                r[0] += np.sum(pos_rewards_a0) + np.sum(neg_rewards_a0) + transfer
                r[1] += np.sum(pos_rewards_a1) + np.sum(neg_rewards_a1) - transfer

            info['contracting'] = 1
            return observations, r, done, info

    def get_q_vals(self, observation, task_prio):
        return self.valuation_nets[task_prio].compute_q_values(observation)

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

    policy_random = True
    episodes = 10
    episode_steps = 100

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       priorities=params.priorities,
                       contracting=params.contracting,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks,
                       observation=1
                       )

    observation_shape = list(env.observation_space.shape)
    number_of_actions = env.action_space.n

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
        policy_net.load_weights('/Users/kyrill/Documents/research/contracting-agents/-weights.0.pth')

    contract = Contract(policy_net=policy_net,
                        valuation_nets=[policy_net, policy_net],
                        contracting_target_update=params.contracting_target_update,
                        gamma=params.gamma,
                        nb_contracting_steps=params.nb_contracting_steps,
                        mark_up=params.mark_up,
                        render=True)

    combined_frames = []
    for i_episode in range(episodes):

        observations = env.reset()
        observations = deepcopy(observations)
        episode_rewards = np.zeros(params.nb_agents)
        episode_contracts = 0

        combined_frames = drawing_util.render_combined_frames(combined_frames, env, [0, 0], 0, [0, 0])

        for i_step in range(episode_steps):

            actions = []
            for i_ag, agent in enumerate([0, 1]):
                if not env.agents[i_ag].done:
                    if not policy_random:
                        actions.append(agent.forward(observations[i_ag]))
                    else:
                        actions.append(np.random.randint(0, env.nb_actions))
                else:
                    actions.append(0)

            observations, r, done, info = contract.contracting_n_steps(env, observations, actions, combined_frames)


            episode_contracts += info['contracting']
            # TODO: Abdisskontieren
            episode_rewards += r


            combined_frames = drawing_util.render_combined_frames(combined_frames, env, r, 0, actions)

            if all([agent.done for agent in env.agents]):
                break

        print("Episode {} contracts: {}".format(i_episode, episode_contracts))

    export_video('Smart-Factory-Contracting.mp4', combined_frames, None)

if __name__ == '__main__':
    main()
