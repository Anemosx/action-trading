import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
from keras.layers import Dense, Activation, Flatten, Convolution2D

from copy import deepcopy

from common_utils.utils import export_video
import os
from collections import deque
import pandas as pd


def build_agent(params, processor):
    # input_shape = (84, 84, 3)
    input_shape = params.input_shape
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(params.nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    memory = SequentialMemory(limit=params.memory_limit, window_length=1)
    # processor = MABargainingProcessor()

    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=params.eps_val_max,
                                  value_min=params.eps_val_min, value_test=params.eps_val_test,
                                  nb_steps=params.nb_steps)

    dqn = DQNAgent(model=model, nb_actions=params.nb_actions, policy=policy, memory=memory,
                   processor=processor, nb_steps_warmup=params.nb_steps_warmup,
                   gamma=params.gamma, target_model_update=params.target_update,
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    return dqn


def fit_n_agents(env, nb_steps, agents=None, nb_max_episode_steps=None, logger=None, log_dir=None, contract=None):

    for agent in agents:
        if not agent.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet.'
                ' Please call `compile()` before `fit()`.')

        agent.training = True
        agent._on_train_begin()

    episode = 0
    observations = [None for _ in agents]
    episode_rewards = [None for _ in agents]
    episode_steps = [None for _ in agents]

    plan = []
    episode_compensations = np.zeros(2)
    contracting = False

    for agent in agents:
        agent.step = 0
    did_abort = False
    try:
        while agents[0].step < nb_steps:
            if observations[0] is None:  # start of a new episode
                observations = deepcopy(env.reset())
                for i, agent in enumerate(agents):
                    episode_steps[i] = 0
                    episode_rewards[i] = 0.
                    plan = []
                    nb_contracts = 0
                    episode_compensations = np.zeros(2)
                    # Obtain the initial observation by resetting the environment.
                    agent.reset_states()
                    if agent.processor is not None:
                        observations[i] = agent.processor.process_observation(observations[i])
                    assert observations[i] is not None
                    # At this point, we expect to be fully initialized.
                    assert episode_rewards[i] is not None
                    assert episode_steps[i] is not None
                    assert observations[i] is not None

            actions = []
            if len(plan) > 0:
                plan_action = plan.popleft()
                actions.append(plan_action[0])
                actions.append(plan_action[1])
                agents[0].recent_action = actions[0]
                agents[1].recent_action = actions[1]
                contracting = True
            else:
                for i, agent in enumerate(agents):
                    # Run a single step.
                    # This is were all of the work happens. We first perceive and compute the action
                    # (forward step) and then use the reward to improve (backward step).
                    actions.append(agent.forward(observations[i]))
                    if agent.processor is not None:
                        actions[i] = agent.processor.process_action(actions[i])
                contracting = False

            env.contract = contracting

            accumulated_info = {}
            done = False
            '''
            add contracting code here
                Input: env, actions, agents, current observations
                Output: Contracting plan, i.e., list of n actions to be executed
            '''
            if contract is not None:
                if len(plan) == 0:
                    plan, compensations = contract.find_contract(env=env,
                                                                 actions=actions,
                                                                 observations=observations)
                    plan = deque(plan)
                    if len(plan) > 0:
                        nb_contracts += 1
                    episode_compensations += compensations

            observations, r, done, info = env.step(actions)
            observations = deepcopy(observations)

            if contract is not None:
                r, episode_compensations = contract.get_compensated_rewards(agents=agents,
                                                                            rewards=r,
                                                                            episode_compensations=episode_compensations)

            for i, agent in enumerate(agents):
                if agent.processor is not None:
                    observations[i], r[i], done, info = agent.processor.process_step(observations[i], r[i], done, info)

            if nb_max_episode_steps and episode_steps[0] >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True

            for i, agent in enumerate(agents):
                metrics = agent.backward(r[i], terminal=done)
                episode_rewards[i] += r[i]
                episode_steps[i] += 1
                agent.step += 1

            if done:
                for i, agent in enumerate(agents):
                    agent.forward(observations[i])
                    agent.backward(0., terminal=False)

                logger.write_log('episode_return', np.sum(episode_rewards), episode)
                for i, agent in enumerate(agents):
                    logger.write_log('episode_return_agent-{}'.format(i), episode_rewards[i], episode)
                logger.write_log('episode_compensations_a1', episode_compensations[0], episode)
                logger.write_log('episode_compensations_a2', episode_compensations[1], episode)
                logger.write_log('number_contracts', nb_contracts, episode)
                logger.write_log('episode_steps', episode_steps[0], episode)
                #for key, value in info.items():
                #    logger.write_log(key, value, agents[0].step)

                observations = [None for _ in agents]
                episode_steps = [None for _ in agents]
                episode_rewards = [None for _ in agents]
                episode_compensations = np.zeros(2)
                nb_contracts = 0
                episode += 1

    except KeyboardInterrupt:
        # We catch keyboard interrupts here so that training can be be safely aborted.
        # This is so common that we've built this right into this function, which ensures that
        # the `on_train_end` method is properly called.
        did_abort = True
    for i, agent in enumerate(agents):
        agent._on_train_end()


def fit_n_agents_n_step_contracting(env,
                                    nb_steps,
                                    agents=None,
                                    nb_max_episode_steps=None,
                                    logger=None,
                                    contract=None):

    for agent in agents:
        if not agent.compiled:
            raise RuntimeError(
                'Your tried to fit your agent but it hasn\'t been compiled yet.'
                ' Please call `compile()` before `fit()`.')

        agent.training = True
        agent._on_train_begin()

    episode = 0
    observations = [None for _ in agents]
    episode_rewards = [None for _ in agents]
    episode_steps = [None for _ in agents]

    episode_compensations = np.zeros(2)
    episode_contracts = 0

    for agent in agents:
        agent.step = 0
    did_abort = False
    try:
        while agents[0].step < nb_steps:
            if observations[0] is None:  # start of a new episode
                observations = deepcopy(env.reset())
                for i, agent in enumerate(agents):
                    episode_steps[i] = 0
                    episode_rewards[i] = 0.
                    episode_contracts = 0
                    episode_compensations = np.zeros(2)
                    # Obtain the initial observation by resetting the environment.
                    agent.reset_states()
                    if agent.processor is not None:
                        observations[i] = agent.processor.process_observation(observations[i])
                    assert observations[i] is not None
                    # At this point, we expect to be fully initialized.
                    assert episode_rewards[i] is not None
                    assert episode_steps[i] is not None
                    assert observations[i] is not None

            actions = []
            for i, agent in enumerate(agents):
                # Run a single step.
                # This is were all of the work happens. We first perceive and compute the action
                # (forward step) and then use the reward to improve (backward step).
                actions.append(agent.forward(observations[i]))
                if agent.processor is not None:
                    actions[i] = agent.processor.process_action(actions[i])

            contracting = False
            if contract is not None:
                contracting, greedy = contract.check_contracting(env, actions)

            accumulated_info = {}
            done = False

            env.contract = contracting
            if contracting:
                observations, r, done, info, compensation = contract.contracting_n_steps(env, observations, greedy)
                episode_compensations += compensation
                episode_contracts += 1
            else:
                observations, r, done, info = env.step(actions)
                observations = deepcopy(observations)

            if contract is not None:
                r, episode_compensations = contract.get_compensated_rewards(agents=agents,
                                                                        rewards=r,
                                                                        episode_compensations=episode_compensations)

            for i, agent in enumerate(agents):
                if agent.processor is not None:
                    observations[i], r[i], done, info = agent.processor.process_step(observations[i], r[i], done, info)

            if nb_max_episode_steps and episode_steps[0] >= nb_max_episode_steps - 1:
                # Force a terminal state.
                done = True

            for i, agent in enumerate(agents):
                metrics = agent.backward(r[i], terminal=done)
                episode_rewards[i] += r[i]
                episode_steps[i] += 1
                agent.step += 1

            if done:
                for i, agent in enumerate(agents):
                    agent.forward(observations[i])
                    agent.backward(0., terminal=False)

                logger.write_log('episode_return', np.sum(episode_rewards), episode)
                for i, agent in enumerate(agents):
                    logger.write_log('episode_return_agent-{}'.format(i), episode_rewards[i], episode)
                logger.write_log('episode_compensations_a1', compensation[0], episode)
                logger.write_log('episode_compensations_a2', compensation[1], episode)
                logger.write_log('contracting', int(episode_contracts), episode)
                logger.write_log('a1_greedy', int(greedy[0]), episode)
                logger.write_log('a2_greedy', int(greedy[1]), episode)
                logger.write_log('episode_steps', episode_steps[0], episode)
                #for key, value in info.items():
                #    logger.write_log(key, value, agents[0].step)

                observations = [None for _ in agents]
                episode_steps = [None for _ in agents]
                episode_rewards = [None for _ in agents]
                episode_compensations = np.zeros(2)
                episode_contracts = 0
                episode += 1


    except KeyboardInterrupt:
        # We catch keyboard interrupts here so that training can be be safely aborted.
        # This is so common that we've built this right into this function, which ensures that
        # the `on_train_end` method is properly called.
        did_abort = True
    for i, agent in enumerate(agents):
        agent._on_train_end()


def test_n_agents(env, agents=[], nb_episodes=1, nb_max_episode_steps=None, log_dir=None, log_episode=None, contract=None):

    for i, agent in enumerate(agents):
        if not agent.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet.'
                               ' Please call `compile()` before `test()`.')

        agent.training = False
        agent.step = 0
        agent._on_test_begin()

    frames = []

    for episode in range(nb_episodes):
        episode_rewards = [0. for _ in agents]
        episode_step = 0
        nb_contracts = 0
        plan = []
        episode_compensations = np.zeros(2)
        observations = deepcopy(env.reset())
        contracting = False

        best_q_in_s = []
        if contract is not None:
            for i in range(2):
                best_q_in_s.append(np.max(contract.agents[i].compute_q_values(observations[i])))
        else:
            for i in range(2):
                best_q_in_s.append(np.max(agents[i].compute_q_values(observations[i])))

        info_values = [{'reward': 0.0,
                        'agent_ep_comp': episode_compensations[i],
                        'action': -1,
                        'contracting': len(plan),
                        'ask': -1,
                        'bid': -1,
                        'best_q_in_s': np.max(agents[i].compute_q_values(observations[i]))
                        } for i in range(env.nb_agents)]
        frames.append(env.render(mode='rgb_array', info_values=info_values))

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
                contracting = True
            else:
                for i, agent in enumerate(agents):
                    actions.append(agent.forward(observations[i]))
                    if agent.processor is not None:
                        actions[i] = agent.processor.process_action(actions[i])
                contracting = False

            '''
            add contracting code here
            Input: env, actions, agents, current observations
            Output: Contracting plan, i.e., list of n actions to be executed
            '''
            env.contract = contracting
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

            best_q_in_s = []
            if contract is not None:
                for i in range(2):
                    best_q_in_s.append(np.max(contract.agents[i].compute_q_values(observations[i])))
            else:
                for i in range(2):
                    best_q_in_s.append(np.max(agents[i].compute_q_values(observations[i])))

            for i, agent in enumerate(env.agents):
                info_values[i]['reward'] = r[i]
                info_values[i]['agent_ep_comp'] = episode_compensations[i]
                info_values[i]['action'] = actions[i]
                info_values[i]['best_q_in_s'] = best_q_in_s[i]
                info_values[i]['contracting'] = len(plan)
                info_values[i]['ask'] = env.actions[actions[i]][3]
                info_values[i]['bid'] = env.actions[actions[i]][2]

            frames.append(env.render(mode='rgb_array', info_values=info_values))

            for i, agent in enumerate(agents):
                if agent.processor is not None:
                    observations[i], r[i], done, info = agent.processor.process_step(observations[i], r[i], d, info)

            if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                done = True
            for i, agent in enumerate(agents):
                agent.backward(r[i], terminal=done)
                episode_rewards[i] += r[i]
                agent.step += 1
            episode_step += 1

        # We are in a terminal state but the agent hasn't yet seen it. We therefore
        # perform one more forward-backward call and simply ignore the action before
        # resetting the environment. We need to pass in `terminal=False` here since
        # the *next* state, that is the state of the newly reset environment, is
        # always non-terminal by convention.

        for i, agent in enumerate(agents):
            agent.forward(observations[i])
            agent.backward(0., terminal=False)


    export_video(os.path.join(log_dir, 'MA-{}.mp4'.format(log_episode)), frames, None)
    for i, agent in enumerate(agents):
        agent._on_test_end()


def test_n_agents_n_step_contracting(env, agents=[], nb_episodes=1, nb_max_episode_steps=None, log_dir=None, log_episode=None, contract=None):

    for i, agent in enumerate(agents):
        if not agent.compiled:
            raise RuntimeError('Your tried to test your agent but it hasn\'t been compiled yet.'
                               ' Please call `compile()` before `test()`.')

        agent.training = False
        agent.step = 0
        agent._on_test_begin()

    df = pd.DataFrame(columns=('episode',
                               'contracting',
                               'reward',
                               'reward_a1', 'reward_a2',
                               'episode_compensations_a1', 'episode_compensations_a2',
                               'number_contracts',
                               'episode_steps'))

    frames = []

    for episode in range(nb_episodes):
        episode_rewards = [0. for _ in agents]
        episode_step = 0
        nb_contracts = 0
        plan = []
        episode_compensations = np.zeros(2)
        observations = deepcopy(env.reset())

        info_values = [{'reward': 0.0,
                        'agent_comp': episode_compensations[i],
                        'contracting': len(plan),
                        'a{}_greedy'.format(i): -1,
                        } for i in range(env.nb_agents)]
        frames.append(env.render(mode='rgb_array', info_values=info_values))

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

            for i, agent in enumerate(agents):
                actions.append(agent.forward(observations[i]))
                if agent.processor is not None:
                    actions[i] = agent.processor.process_action(actions[i])

            contracting = False
            if contract is not None:
                contracting, greedy = contract.check_contracting(env, actions)

            env.contract = contracting
            if contracting:
                observations, r, done, info, compensation = contract.contracting_n_steps(env,
                                                                                         observations,
                                                                                         greedy,
                                                                                         frames,
                                                                                         info_values)
                episode_compensations += compensation
            else:
                observations, r, d, info = env.step(actions)
                observations = deepcopy(observations)

            if contract is not None:
                r, episode_compensations = contract.get_compensated_rewards(agents=agents,
                                                                        rewards=r,
                                                                        episode_compensations=episode_compensations)

            for i, agent in enumerate(env.agents):
                info_values[i]['reward'] = r[i]
                info_values[i]['agent_comp'] = episode_compensations[i]
                info_values[i]['contracting'] = 0
                info_values[i]['a{}_greedy'.format(i)] = greedy[i]

            frames.append(env.render(mode='rgb_array', info_values=info_values))

            for i, agent in enumerate(agents):
                if agent.processor is not None:
                    observations[i], r[i], done, info = agent.processor.process_step(observations[i], r[i], d, info)

            if nb_max_episode_steps and episode_step >= nb_max_episode_steps - 1:
                done = True
            for i, agent in enumerate(agents):
                agent.backward(r[i], terminal=done)
                episode_rewards[i] += r[i]
                agent.step += 1
            episode_step += 1

        # We are in a terminal state but the agent hasn't yet seen it. We therefore
        # perform one more forward-backward call and simply ignore the action before
        # resetting the environment. We need to pass in `terminal=False` here since
        # the *next* state, that is the state of the newly reset environment, is
        # always non-terminal by convention.

        for i, agent in enumerate(agents):
            agent.forward(observations[i])
            agent.backward(0., terminal=False)

        df.loc[episode] = [episode,
                           (contract is not None),
                           episode_rewards[0] + episode_rewards[1],
                           episode_rewards[0], episode_rewards[1],
                           episode_compensations[0], episode_compensations[1],
                           nb_contracts,
                           episode_step]

    df.to_csv(os.path.join(log_dir, 'test-values-contracting-{}.csv'.format(contract is not None)))

    export_video(os.path.join(log_dir, 'MA-{}.mp4'.format(log_episode)), frames, None)
    for i, agent in enumerate(agents):
        agent._on_test_end()










