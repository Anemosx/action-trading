from datetime import datetime
from scipy import stats
import numpy as np
import matplotlib.pyplot as pl
import pytorch_training



def evaluate(agents, environment, evaluation_episodes: int, steps_per_episode: int, scenario_id: str, log_results: bool,
             plot_results: bool, render_environment: bool, logger=None):
    print("Evaluation of {} started at {}".format(scenario_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    if render_environment:
        raise NotImplemented
    else:
        render_channel, render_stub = None, None

    log = pytorch_training.prepare_log(evaluation_episodes)

    for episode in range(0, evaluation_episodes):
        observations = environment.reset()
        done = False
        current_step = 0
        agent_indices = list(range(0, len(environment.agents)))
        episode_steps = 0
        episode_reward = np.zeros(len(environment.agents))

        while not done:

            actions = []
            for agent_index in agent_indices:
                action = agents[agent_index].policy(observations[agent_index])
                actions.append(action)

            next_observations, joint_reward, joint_done, _ = environment.step(actions)

            current_step += 1
            episode_reward += joint_reward
            done = all(done is True for done in joint_done) or current_step == steps_per_episode


            if render_environment:
                environment.render(render_stub)

        if logger is not None:
            logger.log_metric('episode_return', np.sum(episode_reward))
            logger.log_metric('episode_steps', episode_steps)

        print("Episode reward - {}".format(np.sum(episode_reward)))
        # print progress every now and then
        if episode > 0 and episode % 100 is 0:
            recent_rewards = log["shaped reward"][episode - 100:episode]
            fl_avg_rew = sum(recent_rewards) / len(recent_rewards)
            print("episode: {}, reward (fl.avg.): {:.3f}".format(episode, fl_avg_rew))

        # buffer results
        # pytorch_training.add_log_entry(log, environment, episode)

    if render_environment:
        render_channel.close()

    print("Evaluation of {} finished at {}".format(scenario_id, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("reward: μ={}, σ={} ".format(
        np.around(np.average(log["shaped reward"]), decimals=2), np.around(np.std(log["shaped reward"]), decimals=2)))

    if log_results:
        data.save_csv("results/evaluation/{}.csv".format(scenario_id), log)

    if plot_results:
        pl.rcdefaults()
        fig, ax = pl.subplots()

        bars = ('reward', 'steps', 'safety\nviolations')
        y_pos = np.arange(len(bars))
        quantities = [np.average(log["shaped reward"]),
                      np.average(log["steps until solved"]),
                      np.average(log["wrong enqueueings"] + log["boundary collisions"] + log["agent collisions"])]
        errors = [np.std(log["shaped reward"]),
                  np.std(log["steps until solved"]),
                  np.std(log["wrong enqueueings"] + log["boundary collisions"] + log["agent collisions"])]
        ax.barh(y_pos,
                quantities,
                xerr=errors,
                align='center',
                alpha=0.7,
                ecolor='black',
                capsize=7,
                label='performance')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bars)
        # ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Performance')
        ax.set_title('Evaluation of ' + scenario_id)

        pl.show()


def get_mean_and_ci(data_series, confidence=0.997):
    a = 1.0 * np.array(data_series)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    # simplified for plotting, original:
    # return ci as tupel of (lower bound, upper bound) -> (m-h, m+h)
    return m, 2 * h


def get_median_and_perc(data_series, percentile=95.0):
    samples = 1.0 * np.array(data_series)
    median, percentile = np.median(samples), np.percentile(samples, percentile)
    print(median, percentile)
    return median, percentile


def plot_mean_and_ci(data_series, labels):
    # blue bars with error
    shaped_reward, shaped_reward_ci = list(), list()
    # cyan bars with error
    steps_until_solved, steps_until_solved_ci = list(), list()
    # grey bars with error
    safety_violations, safety_violations_ci = list(), list()

    for reward_function_id in range(0, len(labels)):
        avg, ci = get_mean_and_ci(data_series[reward_function_id]["shaped reward"])
        shaped_reward.append(avg)
        shaped_reward_ci.append(ci)
        avg, ci = get_mean_and_ci(data_series[reward_function_id]["steps until solved"])
        steps_until_solved.append(avg)
        steps_until_solved_ci.append(ci)
        avg, ci = get_mean_and_ci(data_series[reward_function_id]["wrong enqueueings"] +
                                  data_series[reward_function_id]["boundary collisions"] +
                                  data_series[reward_function_id]["agent collisions"])
        safety_violations.append(avg)
        safety_violations_ci.append(ci)

    # The x position of bars
    pos = np.arange(len(labels))

    pl.figure(dpi=300)
    axes = pl.gca()
    axes.set_ylim([-20.0, 55.0])

    pl.errorbar(pos, shaped_reward, yerr=shaped_reward_ci, elinewidth=2, capsize=2, color='xkcd:apple',
                ecolor='black', alpha=0.6, label='score (mean, 99.99% ci)')
    pl.errorbar(pos, steps_until_solved, yerr=steps_until_solved_ci, elinewidth=2, capsize=2, color='xkcd:azure',
                ecolor='black', alpha=0.6, label='mean steps (mean, 99.99% ci)')
    pl.errorbar(pos, safety_violations, yerr=safety_violations_ci, elinewidth=2, capsize=2, color='xkcd:red',
                ecolor='black', alpha=0.6, label='safety violations (mean, 99.99% ci)')

    # general layout
    pl.xticks([r for r in range(len(labels))], labels)
    pl.ylabel('cumulative score')
    pl.xlabel('reward function')
    pl.legend()
    pl.grid(True)

    # Show graphic
    pl.show()
    # pl.savefig("/results/a_result.png")


def plot_mean_and_perc(data_series, labels):
    # blue bars with error
    score_median, score_percentile = list(), list()
    # cyan bars with error
    steps_median, steps_percentile = list(), list()
    # grey bars with error
    violations_median, violations_percentile = list(), list()

    for reward_function_id in range(0, len(labels)):
        median, percentile = get_median_and_perc(data_series[reward_function_id]["shaped reward"], 0.03)
        score_median.append(median)
        score_percentile.append(percentile)
        median, percentile = get_median_and_perc(data_series[reward_function_id]["steps until solved"], 99.7)
        steps_median.append(median)
        steps_percentile.append(percentile)
        median, percentile = get_median_and_perc(data_series[reward_function_id]["wrong enqueueings"] +
                                                 data_series[reward_function_id]["boundary collisions"] +
                                                 data_series[reward_function_id]["agent collisions"], 99.7)
        violations_median.append(median)
        violations_percentile.append(percentile)

    # width of the bars
    bar_width = 0.15

    # The x position of bars
    r1 = np.arange(len(labels))
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2 * bar_width for x in r1]
    r4 = [x + 3 * bar_width for x in r1]
    r5 = [x + 4 * bar_width for x in r1]
    r6 = [x + 5 * bar_width for x in r1]

    pl.figure(dpi=300)
    axes = pl.gca()
    axes.set_ylim([-20.0, 210.0])

    pl.bar(r1, score_median, width=bar_width, color='xkcd:apple', alpha=0.6, label='score: median')
    pl.bar(r2, score_percentile, width=bar_width, color='xkcd:apple', label='score: 0.03 percentile')
    pl.bar(r3, steps_median, width=bar_width, color='xkcd:azure', alpha=0.6, label='steps: median')
    pl.bar(r4, steps_percentile, width=bar_width, color='xkcd:azure', label='steps: 99.7 percentile)')
    pl.bar(r5, violations_median, width=bar_width, color='xkcd:marigold', alpha=0.6, label='safety violations: median')
    pl.bar(r6, violations_percentile, width=bar_width, color='xkcd:marigold', label='safety violations: 99.7 percentile')

    # general layout
    pl.xticks([r + bar_width for r in range(len(labels))], labels)
    pl.ylabel('cumulative score')
    pl.xlabel('reward function')
    pl.legend(ncol=2)
    pl.grid(True)

    # Show graphic
    pl.show()
    # pl.savefig("/results/a_result.png")
