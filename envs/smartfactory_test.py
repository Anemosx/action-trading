from envs.smartfactory import Smartfactory
import sys
from random import randint
from dotmap import DotMap
import json


def test_random_agent(num_episodes=1):

    with open('params.json', 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)


    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       priorities=params.priorities,
                       contracting=0,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks
                       )
    episodes = 0

    while episodes < num_episodes:
        try:
            env.reset()
            done = False
            episode_steps = 0
            while not done:
                    action_1 = randint(0, 3)
                    action_2 = randint(0, 3)
                    observations, rewards, done, info = env.step([action_1, action_2])
                    # env_mem_size = sys.getsizeof(env)
                    # print('Episode:', episodes, 'Episode Steps:', episode_steps, 'Env_Mem_Size:', env_mem_size)
                    episode_steps += 1
            episodes += 1
        except KeyboardInterrupt:
            break


if __name__ == '__main__':
    test_random_agent(num_episodes=1000)
