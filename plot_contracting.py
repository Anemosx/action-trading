import os
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

if __name__ == '__main__':

    log_dir = 'experiments/20190914-13-44-15'
    nb_runs = 8
    window = 20
    episodes = 2500
    runs = [0, 2]
    testing = False

    i = 0
    for run in range(nb_runs):
        for c in runs:

            df_new = pd.read_csv(os.path.join(log_dir,
                                          'run-{}'.format(run),
                                          'contracting-{}-mark-up-1.4'.format(c),
                                          'train-values-contracting-{}.csv'.format(c>0)))

            df_new = df_new[:episodes]
            df_new = df_new.rolling(window=window,center=False).mean()

            if i > 0:
                df = df.append(df_new, ignore_index=True)
            elif i == 0:
                df = df_new.copy()

            if testing:

                df_new_test = pd.read_csv(os.path.join(log_dir,
                                                       'run-{}'.format(run),
                                                       'contracting-{}'.format(c),
                                                       'test-values-contracting-{}.csv'.format(c)))
                if i > 0:
                    df_test = df_test.append(df_new_test, ignore_index=True)
                elif i == 0:
                    df_test = df_new_test.copy()

            i += 1


    sns.lineplot(x="episode", y="reward", hue="contracting", data=df)
    plt.savefig(os.path.join(log_dir, 'reward.png'))
    plt.clf()

    sns.lineplot(x="episode", y="reward_a1",hue="contracting", data=df)
    plt.savefig(os.path.join(log_dir, 'reward_a1.png'))
    plt.clf()

    sns.lineplot(x="episode", y="reward_a2", hue="contracting", data=df)
    plt.savefig(os.path.join(log_dir, 'reward_a2.png'))
    plt.clf()

    df_a1_greedy = df[['episode', 'greedy_a1', 'contracting']]
    df_a1_greedy['agent'] = 'a1'
    df_a1_greedy.rename(columns={'greedy_a1': 'greedy'}, inplace=True)

    df_a2_greedy = df[['episode','greedy_a2', 'contracting']]
    df_a2_greedy['agent'] = 'a2'
    df_a2_greedy.rename(columns={'greedy_a2': 'greedy'}, inplace=True)

    greedy = df_a1_greedy.append(df_a2_greedy, ignore_index=True)
    contracting = greedy['contracting'] == True

    sns.lineplot(x="episode", y="greedy", hue="agent", markers=True, dashes=False, data=greedy[contracting])
    plt.savefig(os.path.join(log_dir, 'greedy_policy_usage.png'))
    plt.clf()

    """
    sns.lineplot(x="episode", y="accumulated_transfer_a1", data=df[contracting])
    plt.savefig(os.path.join(log_dir, 'accumulated_transfer_a1.png'))
    plt.clf()

    sns.lineplot(x="episode", y="accumulated_transfer_a2", data=df[contracting])
    plt.savefig(os.path.join(log_dir, 'accumulated_transfer_a2.png'))
    plt.clf()

    contracting = df['contracting'] == True
    sns.lineplot(x="episode", y="number_contracts", data=df[contracting])
    plt.savefig(os.path.join(log_dir, 'number_contracts.png'))
    plt.clf()

    sns.lineplot(x="episode", y="episode_steps", hue="contracting", data=df)
    plt.savefig(os.path.join(log_dir, 'episode_steps.png'))
    plt.clf()

    sns.lineplot(x="episode", y="number_contracts", data=df[contracting])
    plt.savefig(os.path.join(log_dir, 'number_contracts.png'))
    plt.clf()

    if testing:
        sns.boxplot(x="contracting", y="reward",
                    data=df_test, palette="Set3")
        plt.savefig(os.path.join(log_dir, 'rewards-var.png'))
        plt.clf()
    
        sns.boxplot(x="contracting", y="reward_a1",
                    data=df_test, palette="Set3")
        plt.savefig(os.path.join(log_dir, 'rewards_a1-var.png'))
        plt.clf()
    
        sns.boxplot(x="contracting", y="reward_a2",
                    data=df_test, palette="Set3")
        plt.savefig(os.path.join(log_dir, 'rewards_a2-var.png'))
        plt.clf()
    
    sns.distplot(df[['action_a1']])
    plt.savefig(os.path.join(log_dir, 'action_a1.png'))
    plt.clf()

    sns.distplot(df[['action_a2']])
    plt.savefig(os.path.join(log_dir, 'action_a2.png'))
    plt.clf()
    
    """