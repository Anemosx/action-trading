from datetime import datetime
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
from copy import deepcopy

def plot_score(data_frame: pd.DataFrame, window_size: int):
    # compute moving average and standard deviation
    score_avg = data_frame["steps until solved"].rolling(window=window_size).mean().squeeze()
    score_std = data_frame["steps until solved"].rolling(window=window_size).std().squeeze()
    steps_avg = data_frame["shaped reward"].rolling(window=window_size).mean().squeeze()
    steps_std = data_frame["shaped reward"].rolling(window=window_size).std().squeeze()
    # create plot
    pl.figure(dpi=150)
    pl.plot(score_avg.index, score_avg, color='black', label="steps until solved")
    pl.fill_between(score_avg.index, score_avg - score_std, score_avg + score_std, color='black', alpha=0.2)
    pl.plot(steps_avg.index, steps_avg, color='C0', label="shaped reward")
    pl.fill_between(steps_avg.index, steps_avg - steps_std, steps_avg + steps_std, color='C0', alpha=0.2)
    pl.legend(loc='upper right')
    pl.title("task completion during training")
    pl.xlabel("training episode")
    pl.ylabel("cumulative reward")
    pl.show()


def plot_violations(data_frame: pd.DataFrame, window_size: int):
    #  compute moving average
    data_series_r = data_frame["wrong enqueueings"].rolling(window=window_size).mean().squeeze()
    data_series_g = data_frame["boundary collisions"].rolling(window=window_size).mean().squeeze()
    data_series_b = data_frame["agent collisions"].rolling(window=window_size).mean().squeeze()
    # create plot
    pl.figure(dpi=150)
    pl.plot(data_series_r.index, data_series_r, 'black', label="wrong enqueueings")
    pl.plot(data_series_g.index, data_series_g, 'C0', label="boundary collisions")
    pl.plot(data_series_b.index, data_series_b, 'C9', label="agent collisions")
    pl.legend(loc='upper right')
    pl.title("safety violations during training")
    pl.xlabel("training episode")
    pl.ylabel("cumulative safety violations")
    pl.show()


def train_dqn(agents, environment, training_episodes: int, steps_per_episode: int, scenario_id: str,
              logger, plot_training_progress: bool, contract):
    print("{} | {} | training started".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), scenario_id))

    log = prepare_log(training_episodes)

    for episode in range(0, training_episodes):
        observations = environment.reset()
        done = False
        current_step = 0
        agent_indices = list(range(0, len(environment.agents)))
        episode_steps = 0
        episode_return = np.zeros(len(environment.agents))
        episode_contracts = 0
        joint_done = [False, False]

        while not done:

            actions = []
            for agent_index in agent_indices:
                if not joint_done[agent_index]:
                    action = agents[agent_index].policy(observations[agent_index])
                else:
                    action = np.random.randint(0, 4)
                actions.append(action)

            next_observations, joint_reward, joint_done, info = contract.contracting_n_steps(environment, observations, actions)

            # buffer experience
            for agent_index in agent_indices:
                agents[agent_index].save(observations[agent_index],
                                         actions[agent_index],
                                         next_observations[agent_index],
                                         joint_reward[agent_index],
                                         joint_done[agent_index])

            for agent_index in agent_indices:
                if not joint_done[agent_index]:
                    # train the brain
                    agents[agent_index].train()

            observations = next_observations

            # finish current step
            current_step += 1
            episode_steps += 1
            episode_return += joint_reward
            episode_contracts += info['contracting']

            done = all(done is True for done in joint_done) or current_step == steps_per_episode

        if logger is not None:
            logger.log_metric('episode_return', np.sum(episode_return))
            logger.log_metric('episode_steps', episode_steps)
            logger.log_metric('episode_contracts', episode_contracts)
        # buffer results
        # add_log_entry(log, environment, episode)

        # print progress every now and then
        if episode > 0 and episode % 25 is 0:
            recent_rewards = log["shaped reward"][episode-25:episode]
            fl_avg_rew = sum(recent_rewards) / len(recent_rewards)
            for agent in agents:
                print("episode: {}, epsilon: {:.5f}, reward (fl.avg.): {:.3f}".format(episode, agent.epsilon, fl_avg_rew))

    #if log_training_progress:
    #    data.save_csv("results/{}.csv".format(scenario_id), log)

    if plot_training_progress:
        plot_score(data_frame=pd.DataFrame(log), window_size=25)
        plot_violations(data_frame=pd.DataFrame(log), window_size=50)

    print("{} | {} | training finished".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), scenario_id))


def train_trading_dqn(agents, environment, training_episodes: int, steps_per_episode: int, logger, trade, trading_mode, trading_budget):
    print("{} | training started".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    log = prepare_log(training_episodes)

    for episode in range(0, training_episodes):
        observations = environment.reset()
        done = False
        current_step = 0
        agent_indices = list(range(0, len(environment.agents)))
        episode_return = np.zeros(len(environment.agents))
        joint_done = [False, False]
        trade.trading_budget = deepcopy(trading_budget)
        trade_count = np.zeros(len(agents))
        accumulated_transfer = np.zeros(len(agents))

        while not done:

            actions = []
            for agent_index in agent_indices:
                if not joint_done[agent_index]:
                    action = agents[agent_index].policy(observations[agent_index])
                else:
                    action = np.random.randint(0, 4)
                actions.append(action)

            joint_reward, next_observations, joint_done, new_trades, act_transfer = trade.trading_step(episode_return, environment, actions)

            # buffer experience
            for agent_index in agent_indices:
                agents[agent_index].save(observations[agent_index],
                                         actions[agent_index],
                                         next_observations[agent_index],
                                         joint_reward[agent_index],
                                         joint_done[agent_index])

            for agent_index in agent_indices:
                if not joint_done[agent_index]:
                    # train the brain
                    agents[agent_index].train()

            observations = next_observations
            current_step += 1

            for i in range(len(trade.agents)):
                episode_return[i] += joint_reward[i]
                trade_count[i] += new_trades[i]
                accumulated_transfer[i] += act_transfer[i]

            done = all(done is True for done in joint_done) or current_step == steps_per_episode
            #done = joint_done.__contains__(True) or current_step == steps_per_episode

        if logger is not None:
            logger.log_metric('episode_return', np.sum(episode_return))
            logger.log_metric('episode_steps', current_step)
            logger.log_metric('episode_trades', np.sum(trade_count))
            logger.log_metric('accumulated_transfer', np.sum(accumulated_transfer))
            logger.log_metric('episode_return-0', episode_return[0])
            logger.log_metric('episode_return-1', episode_return[1])
            logger.log_metric('trades-0', trade_count[0])
            logger.log_metric('trades-1', trade_count[1])
            logger.log_metric('accumulated_transfer-0', accumulated_transfer[0])
            logger.log_metric('accumulated_transfer-1', accumulated_transfer[1])

        # print progress every now and then
        if episode > 0 and episode % 25 is 0:
            recent_rewards = log["shaped reward"][episode-25:episode]
            fl_avg_rew = sum(recent_rewards) / len(recent_rewards)
            for agent in agents:
                print("episode: {}, epsilon: {:.5f}, reward (fl.avg.): {:.3f}".format(episode, agent.epsilon, fl_avg_rew))

    print("{} | training finished".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def prepare_log(training_episodes: int):
    return {"steps until solved": [0] * training_episodes,
            "shaped reward": [0] * training_episodes,
            "wrong enqueueings": [0] * training_episodes,
            "boundary collisions": [0] * training_episodes,
            "agent collisions": [0] * training_episodes}


def add_log_entry(log, env, episode):
    log["steps until solved"][episode] = env.time_step
    for ag in env.agents:
        log["shaped reward"][episode] += ag.get_shaped_reward()
        log["wrong enqueueings"][episode] += ag.wrong_enqueueings
        log["boundary collisions"][episode] += ag.boundary_collisions
        log["agent collisions"][episode] += ag.agent_collisions
