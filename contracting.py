import os
import numpy as np
from copy import deepcopy
from envs.smartfactory import Smartfactory
from common_utils.utils import export_video
from agent import build_agent
from dotmap import DotMap
import json
import common_utils.drawing_util as drawing_util


class Contract:

    def __init__(self,
                 policy_net,
                 valuation_nets,
                 contracting_target_update=0,
                 nb_contracting_steps=10,
                 mark_up=1.0,
                 render=True):

        self.policy_net = policy_net
        self.valuation_nets = valuation_nets
        self.contracting_target_update = contracting_target_update
        self.nb_contracting_steps = nb_contracting_steps
        self.mark_up = mark_up
        self.render = render

    def contracting_n_steps(self, env, observations, actions, combined_frames=None):

        contracting, greedy = env.check_contracting(actions)

        if not contracting:

            observations, r, done, info = env.step(actions)

            observations = deepcopy(observations)
            for i in range(2):
                if self.policy_net.processor is not None:
                    observations[i] = self.policy_net.processor.process_observation(observations[i])

            return observations, r, done, info

        else:

            rewards = np.zeros(2)
            done = False
            info = None

            priorities = env.priorities

            if priorities[0]:
                # high prio
                q_vals_a1 = self.valuation_nets[1].compute_q_values(observations[0])
            else:
                # low prio
                q_vals_a1 = self.valuation_nets[0].compute_q_values(observations[0])

            if priorities[1]:
                # high prio
                q_vals_a2 = self.valuation_nets[1].compute_q_values(observations[1])
            else:
                # low prio
                q_vals_a2 = self.valuation_nets[0].compute_q_values(observations[1])

            q_vals = [q_vals_a1, q_vals_a2]

            for i_agent in range(2):
                if not greedy[i_agent]:
                    transfer = np.maximum(np.max(q_vals[i_agent]), 0) * self.mark_up
                    env.agents[(i_agent + 1) % 2].episode_debts += transfer

            for c_step in range(self.nb_contracting_steps):

                c_actions = []
                for i_agent in range(2):
                    if greedy[i_agent]:
                        c_actions.append(self.policy_net.forward(observations[i_agent]))
                    else:
                        # TODO: argmin for next q_values should be taken
                        c_actions.append(np.argmin(q_vals[i_agent]))

                observations, r, done, info = env.step(c_actions)

                observations = deepcopy(observations)
                for i in range(2):
                    if self.policy_net.processor is not None:
                        observations[i] = self.policy_net.processor.process_observation(observations[i])

                r, transfer = self.get_compensated_rewards(env=env, rewards=r)
                rewards += r

                if any([agent.done for agent in env.agents]):
                    break

                if self.render and combined_frames is not None and c_step < self.nb_contracting_steps - 1:
                    combined_frames = drawing_util.render_combined_frames(combined_frames, env, r, 1, actions)

            return observations, rewards, done, info

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

    c = 2
    policy_random = False
    episodes = 10
    episode_steps = 100

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       priorities=params.priorities,
                       contracting=c,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks
                       )

    processor = env.SmartfactoryProcessor()

    policy_net = build_agent(params=params, nb_actions=4, processor=processor)
    policy_net.load_weights('experiments/20191105-20-36-43/run-0/contracting-0/dqn_weights-agent-0.h5f')

    valuation_net_low_prio = build_agent(params=params, nb_actions=4, processor=processor)
    valuation_net_low_prio.load_weights('experiments/20191106-11-32-13/run-0/contracting-0/dqn_weights-agent-0.h5f')

    valuation_net_high_prio = build_agent(params=params, nb_actions=4, processor=processor)
    valuation_net_high_prio.load_weights('experiments/20191106-11-30-59/run-0/contracting-0/dqn_weights-agent-0.h5f')

    valuation_nets = [valuation_net_low_prio, valuation_net_high_prio]

    contract = Contract(policy_net=policy_net,
                        valuation_nets=valuation_nets,
                        contracting_target_update=params.contracting_target_update,
                        nb_contracting_steps=params.nb_contracting_steps,
                        mark_up=params.mark_up,
                        render=False)

    agents = []
    for i_agent in range(params.nb_agents):
        agent = build_agent(params=params, nb_actions=env.nb_contracting_actions, processor=processor)
        agents.append(agent)
        agents[i_agent].load_weights('experiments/20191106-20-02-55/run-0/contracting-2/dqn_weights-agent-{}.h5f'.format(i_agent))

    combined_frames = []
    for i_episode in range(episodes):

        observations = env.reset()
        observations = deepcopy(observations)
        for i, agent in enumerate(agents):
            if agent.processor is not None:
                observations[i] = agent.processor.process_observation(observations[i])

        episode_rewards = np.zeros(params.nb_agents)
        combined_frames = drawing_util.render_combined_frames(combined_frames, env, [0, 0], 0, [0, 0])

        for i_step in range(episode_steps):

            actions = []
            for i_ag, agent in enumerate(agents):
                if not env.agents[i_ag].done:
                    if not policy_random:
                        actions.append(agent.forward(observations[i_ag]))
                    else:
                        actions.append(np.random.randint(0, env.nb_actions))
                else:
                    actions.append(0)

            observations, r, done, info = contract.contracting_n_steps(env, observations, actions, combined_frames)
            observations = deepcopy(observations)
            r, transfer = contract.get_compensated_rewards(env=env, rewards=r)
            episode_rewards += r

            qvals = []
            qvals.append(agents[0].compute_q_values(observations[0]))
            qvals.append(agents[1].compute_q_values(observations[1]))
            combined_frames = drawing_util.render_combined_frames(combined_frames, env, r, 0, actions, qvals=qvals)

            if done: #any([agent.done for agent in env.agents]):
                break

    export_video('Smart-Factory-Contracting.mp4', combined_frames, None)

if __name__ == '__main__':
    main()
