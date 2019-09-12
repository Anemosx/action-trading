import os
from dotmap import DotMap
from agent import build_agent, test_n_agents_n_step_contracting
from envs.smartfactory import Smartfactory
import json
from matplotlib import pyplot as plt
import seaborn as sns
from contracting import Contract


if __name__ == '__main__':

    log_dir = 'logs/20190908-11-19-24'
    nb_episodes = 250
    nb_max_episode_steps = 45

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    dfs = []

    for c in [True, False]:

        env = Smartfactory(nb_agents=2,
                           field_width=params.field_width,
                           field_height=params.field_height,
                           rewards=params.rewards,
                           contracting=c)

        params.nb_actions = env.nb_actions
        processor = env.SmartfactoryProcessor()

        agents = []
        for _ in range(params.nb_learners):
            agent = build_agent(params=params, processor=processor)
            agents.append(agent)
        agents[0].load_weights(os.path.join(log_dir, 'contracting-{}/dqn_weights-agent-0.h5f'.format(c)))
        agents[1].load_weights(os.path.join(log_dir, 'contracting-{}/dqn_weights-agent-1.h5f'.format(c)))

        contract = None
        if c:
            params.nb_actions = 4
            contracting_agents = []
            for i in range(2):
                agent = build_agent(params=params, processor=processor)
                contracting_agents.append(agent)
            contracting_agents[0].load_weights(os.path.join(log_dir, 'contracting-False/dqn_weights-agent-0.h5f'))
            contracting_agents[1].load_weights(os.path.join(log_dir, 'contracting-False/dqn_weights-agent-1.h5f'))
            contract = Contract(agent_1=contracting_agents[0], agent_2=contracting_agents[1])
            params.nb_actions = 12

        df = test_n_agents_n_step_contracting(env,
                                              agents=agents,
                                              nb_episodes=nb_episodes,
                                              nb_max_episode_steps=nb_max_episode_steps,
                                              log_dir=log_dir,
                                              contract=contract,
                                              log_video=False)

        dfs.append(df)

    df = dfs[0].append(dfs[1], ignore_index=True)

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
