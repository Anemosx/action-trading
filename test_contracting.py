import os
from dotmap import DotMap
from agent import build_agent, test_n_agents_n_step_contracting
from envs.smartfactory import Smartfactory
import json
from matplotlib import pyplot as plt
import seaborn as sns
from contracting import Contract
import pandas as pd


if __name__ == '__main__':

    log_dir = 'experiments/20190923-10-58-52'
    nb_episodes = 100
    nb_max_episode_steps = 90

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    dfs = []

    for c in []:

        env = Smartfactory(nb_agents=params.nb_agents,
                           field_width=params.field_width,
                           field_height=params.field_height,
                           rewards=params.rewards,
                           step_penalty=params.step_penalty,
                           contracting=c,
                           nb_machine_types=params.nb_machine_types,
                           nb_tasks=params.nb_tasks
                           )

        processor = env.SmartfactoryProcessor()
        params.nb_actions = env.nb_actions

        agents = []
        for i_ag in range(params.nb_agents):
            agent = build_agent(params=params, nb_actions=env.nb_actions, processor=processor)
            agents.append(agent)
            agents[0].load_weights(os.path.join(log_dir, 'run-0/contracting-{}/dqn_weights-agent-0.h5f'.format(c, i_ag)))

        contract = None
        if c > 0:
            contracting_agents = []
            for i in range(params.nb_agents):
                agent = build_agent(params=params, nb_actions=params.nb_actions_no_contracting_action,
                                    processor=processor)
                agent.load_weights(os.path.join(log_dir, 'run-0/contracting-0/dqn_weights-agent-0.h5f'.format(i)))
                contracting_agents.append(agent)

            contract = Contract(agent_1=contracting_agents[0],
                                agent_2=contracting_agents[1],
                                contracting_target_update=params.contracting_target_update,
                                nb_contracting_steps=params.nb_contracting_steps,
                                mark_up=params.mark_up)

        df = test_n_agents_n_step_contracting(env,
                                         agents=agents,
                                         nb_episodes=nb_episodes,
                                         nb_max_episode_steps=nb_max_episode_steps,
                                         log_dir=log_dir,
                                         contract=contract,
                                         log_video=False)

        dfs.append(df)

    '''
    df_1 = pd.read_csv('test-values-contracting-c-2.csv')
    df_2 = pd.read_csv('test-values-contracting-c-0.csv')

    df = df_1.append(df_2, ignore_index=True)

    df_rewards = df[['reward', 'contracting']]
    df_rewards['agent'] = 'all'

    df_rewards_a0 = df[['reward_a0', 'contracting']]
    df_rewards_a0['agent'] = 'a0'
    df_rewards_a0.columns = ['reward','contracting',  'agent']

    df_rewards_a1 = df[['reward_a1', 'contracting']]
    df_rewards_a1['agent'] = 'a1'
    df_rewards_a1.columns = ['reward', 'contracting', 'agent']

    df_rewards = df_rewards.append(df_rewards_a0, ignore_index=True)
    df_rewards = df_rewards.append(df_rewards_a1, ignore_index=True)

    sns.boxplot(x="agent", y="reward", hue='contracting',
                data=df_rewards, palette="Set3", showmeans=True)
    plt.savefig(os.path.join(log_dir, 'rewards-var.png'))
    plt.clf()
    '''

    df_1 = pd.read_csv('experiments/20190923-10-58-52/run-0/contracting-0/train-values-contracting-False.csv')
    df_1 = df_1.rolling(window=30,center=False).mean()

    df_2 = pd.read_csv('experiments/20190923-10-58-52/run-0/contracting-2/train-values-contracting-True.csv')
    df_2 = df_2.rolling(window=30, center=False).mean()

    df = df_1.append(df_2, ignore_index=True)

    sns.lineplot(x="episode", y="reward", hue='contracting', data=df, palette="Set2")
    plt.savefig(os.path.join(log_dir, 'rewards-var.png'))
    plt.clf()



