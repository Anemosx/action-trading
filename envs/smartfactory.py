import pandas as pd
from common_utils.utils import export_video
from rl.core import Processor

import gym
import gym.spaces
from collections import namedtuple
import common_utils.drawing_util as drawing_util
from common_utils.drawing_util import  render_visual_state
from common_utils.drawing_util import Camera
import numpy as np
import Box2D
from Box2D import b2PolygonShape, b2FixtureDef, b2TestOverlap, b2Transform, b2Rot, b2ChainShape
from PIL import Image
import copy
from copy import deepcopy
import itertools as it
import json
import os
import scipy
from dotmap import DotMap
from agents.pytorch_agents import make_dqn_agent
import agents.pytorch_agents as pta
import trading
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns


INPUT_SHAPE = (84, 84, 1)
joint_learning = 'JOINT_LEARNING'
decentral_learning = 'DECENTRAL_LEARNING'


class Agent:

    AgentState = namedtuple('AgentState', 'position')

    def __init__(self, world, env, index, position, task):
        self.world = world
        self.env = env
        self.index = index
        self.color = env.colors['agent-{}'.format(index)]
        self.old_pos = position
        self.current_pos = position
        self.agent_vertices = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        self.body = self.world.CreateDynamicBody(position=position,
                                                 angle=0,
                                                 angularDamping=0.6,
                                                 linearDamping=3.0,
                                                 shapes=[b2PolygonShape(vertices=self.agent_vertices)],
                                                 shapeFixture=b2FixtureDef(density=0.0))

        self.camera = Camera(pos=deepcopy(position), fov_dims=(6, 6)) #fov_dims=(3, 3))
        self.signalling = False
        self.task = task
        self.episode_debts = 0
        self.done = False

    def process_task(self):
        x = self.body.transform.position.x
        y = self.body.transform.position.y

        task_index = -1
        if [x, y] in self.env.goal_positions:
            index_machine = self.env.goal_positions.index([x, y])
            machine = self.env.goals[index_machine]
            if machine.typ in self.task and machine.inactive <= 0:
                machine.inactive = self.env.nb_steps_machine_inactive
                index_task = self.task.index(machine.typ)
                self.task[index_task] = -1
                task_index = index_task

        return task_index

    def process_task_sequential(self):
        x = self.body.transform.position.x
        y = self.body.transform.position.y

        task_index = -1

        if [x, y] in self.env.goal_positions:

            index_machine = self.env.goal_positions.index([x, y])
            machine = self.env.goals[index_machine]

            if machine.typ == self.task[0] and machine.inactive <= 0:
                # index_task = self.task.index(machine.typ)
                machine.inactive = self.env.nb_steps_machine_inactive
                del self.task[0]
                task_index = 0

        return task_index

    def tasks_finished(self):
        return all(i == -1 for i in self.task)

    def set_signalling(self, action):
        self.signalling = False
        self.env.display_objects['agent-{}'.format(self.index)][1].color = self.color

        if action[3] == 1:
            self.signalling = True
            self.env.display_objects['agent-{}'.format(self.index)][1].color = self.env.colors['signalling']

    def reset(self, position, task):
        self.body.position = position
        self.old_pos = position
        self.current_pos = position

        self.task = task
        self.done = False
        self.episode_debts = 0

    def get_state(self):
        ag_state = Agent.AgentState(position=deepcopy((self.body.position.x, self.body.position.y)))
        return ag_state

    def set_state(self, ag_state):
        self.body.transform = (ag_state.position, 0)


class GridCell:

    def __init__(self, env, index, position, typ):
        self.index = index
        self.vertices = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        self.shape = b2PolygonShape(vertices=self.vertices)
        self.position = position
        self.typ = typ
        if self.typ == 'wall':
            self.color = env.colors['wall']
        elif self.typ == 0:
            self.color = env.colors['machine-0']
        elif self.typ == 1:
            self.color = env.colors['machine-1']
        else:
            self.color = env.colors['field']

        self.inactive = 0

    def reset(self, env, index, display_objects):
        self.color = env.colors['field']
        display_objects['gridcell-{}'.format(index)][1].color = self.color
        self.inactive = 0


class Smartfactory(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    State = namedtuple('Smartfactory', 'agent_states')

    class SmartfactoryProcessor(Processor):
        def process_observation(self, observation):
            assert observation.ndim == 3  # (height, width, channel)
            img = Image.fromarray(observation)
            img = img.resize((INPUT_SHAPE[0], INPUT_SHAPE[1])).convert('L')  # resize and convert to grayscale
            processed_observation = np.array(img).reshape(*INPUT_SHAPE)

            assert processed_observation.shape == INPUT_SHAPE

            return processed_observation.astype('uint8')  # saves storage in experience memory

        def process_state_batch(self, batch):
            # We could perform this processing step in `process_observation`. In this case, however,
            # we would need to store a `float32` array instead, which is 4x more memory intensive than
            # an `uint8` array. This matters if we store 1M observations.
            processed_batch = batch.astype('float32') / 255.
            processed_batch = processed_batch.reshape(len(batch), *INPUT_SHAPE)
            return processed_batch

        def process_reward(self, reward):
            return np.clip(reward, -1., 5.)

    def __init__(self,
                 nb_agents,
                 field_width,
                 field_height,
                 rewards,
                 step_penalties,
                 priorities,
                 learning=decentral_learning,
                 trading_steps=0,
                 trading_actions=None,
                 contracting=0,
                 nb_machine_types=2,
                 nb_steps_machine_inactive=10,
                 nb_tasks=3,
                 observation=0):
        """

        :rtype: observation
        """
        self.world = Box2D.b2World(gravity=(0, 0))
        if observation == 0:
            self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(84, 84, 1))
        elif observation == 1:
            self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(9, field_width, field_height))
        elif observation == 2:
            self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(11, field_width, field_height))
        elif observation == 3:
            self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(12, field_width, field_height))

        self.observation_valuation_space = gym.spaces.Box(0.0, 1.1, shape=(9, field_width, field_height))

        self.velocity_iterations = 6
        self.position_iterations = 2
        self.dt = 1.0 / 15
        self.agent_restitution = 0.5
        self.agent_density = 1.0

        self.colors = {
            'agent-0': (0.20392156862745098, 0.596078431372549, 0.8588235294117647),
            'agent-1': (0.13, 0.15, 0.14, 1.0), #(0.5843137254901961, 0.6470588235294118, 0.6509803921568628, 1.0), # (1.0, 0.85098039215686272, 0.18431372549019609), #
            'outer_field': (0.5843137254901961, 0.6470588235294118, 0.6509803921568628),
            'field': (1.0, 1.0, 1.0, 1.0),
            'wall': (0.5843137254901961, 0.6470588235294118, 0.6509803921568628, 0.4),
            'checkpoint': (0.13, 0.15, 0.14, 1.0),
            'machine-0': (.7, .7, .7, 1.0),  # (0.1803921568627451, 0.8, 0.44313725490196076),  # (0.6078431372549019, 0.34901960784313724, 0.7137254901960784),
            'machine-1': (0.58, 0.64, 0.65, 1.0), # (0.13, 0.15, 0.14, 1.0), # (0.9058823529411765, 0.2980392156862745, 0.23529411764705882),
            'contracting': (0.13, 0.15, 0.14, 1.0),
            'white': (1.0, 1.0, 1.0, 1.0),
            'dark': (0.13, 0.15, 0.14, 1.0),
            'debt_balance': (0.6078431372549019, 0.34901960784313724, 0.7137254901960784)
        }

        self.contracting = contracting

        self.render_mode = False
        self.valuation_training = False

        self.actions = trading_actions
        self.trading_steps = trading_steps
        self.current_suggestions = np.zeros((nb_agents, 1), dtype=int)
        self.missing_suggestions = np.zeros(nb_agents, dtype=int)

        self.actions_log = []
        self.trade_positions = []

        self.learning = learning
        self.nb_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(self.nb_actions)
        self.nb_contracting_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(n=len(self.actions))

        self.info = {'setting': None,
                     'episode': None,
                     'return_a0': None,
                     'return_a1': None
                     }

        self.pixels_per_worldunit = 24
        self.obs_pixels_per_worldunit = 8
        self.camera = Camera(pos=(0, 0), fov_dims=(5, 5)) # fov_dims=(9, 9)
        self.display_objects = dict()
        self.field_width = field_width
        self.field_height = field_height
        self.edge_margin = .2

        # Agents
        self.nb_agents = nb_agents
        self.field_vertices = []
        self.edge_vertices = []
        self.agents = []
        self.market_agents = []
        self.food = []
        self.goals = []
        self.possible_positions = []
        self.wall_positions = []
        self.goal_positions = []
        self.tasks = []
        self.task_positions = []

        self.rewards = rewards
        self.step_penalties = step_penalties
        self.contract = False

        self.nb_machine_types = nb_machine_types
        self.nb_steps_machine_inactive = nb_steps_machine_inactive
        self.nb_tasks = nb_tasks
        self.priorities = []
        self.debt_balances = []
        self.balance = np.zeros(self.nb_agents)
        self.greedy = []
        self.priorities = priorities

    def _create_field(self):

        self.field_vertices = [(-self.field_width/2, -self.field_height/2),
                               (self.field_width/2, -self.field_height/2),
                               (self.field_width/2, self.field_height/2),
                               (-self.field_width/2, self.field_height/2)]
        self.edge_vertices = [(x - self.edge_margin if x < 0 else x + self.edge_margin,
                               y - self.edge_margin if y < 0 else y + self.edge_margin)
                              for (x, y) in self.field_vertices]
        self.field = self.world.CreateBody(shapes=b2ChainShape(vertices=self.edge_vertices))

        self.display_objects['outer_field'] = (-11, drawing_util.Polygon(vertices=self.edge_vertices,
                                                                         body=self.field,
                                                                         color=self.colors['outer_field']))
        self.display_objects['field'] = (-10, drawing_util.Polygon(vertices=self.field_vertices,
                                                                   body=self.field,
                                                                   color=self.colors['field']))

    def reset(self):
        self.world = Box2D.b2World(gravity=(0, 0))
        self.display_objects = dict()
        self.agents = []
        self.market_agents = []
        self.food = []
        self.goals = []
        self.tasks = []
        self.debt_balances = []
        self.possible_positions = [[(-(self.field_width-(self.field_width/2)-1))+column,
                                    (self.field_height/2)-row] for row in range(self.field_height)
                                   for column in range(self.field_width)]

        np_indices = [(x + (self.field_width - (self.field_width / 2) - 1),
                       y + (self.field_height - (self.field_height / 2) - 1)) for x, y in self.possible_positions]

        self.map = {'{}-{}'.format(pp[0],pp[1]): index for pp, index in zip(self.possible_positions, np_indices)}


        field_indices = [pos for pos in range(self.field_width*self.field_height)]
        wall_indices = []
        goal_indices = [0, self.field_width-1, (self.field_width*self.field_height) - self.field_width]
        self.wall_positions = [self.possible_positions[i] for i in wall_indices]
        self.goal_positions = [self.possible_positions[i] for i in goal_indices]
        spawning_positions = list(set(field_indices) - set(wall_indices) - set(goal_indices))
        spawning_indices = np.random.choice(spawning_positions, self.nb_agents, replace=False)

        tasks = [list(np.random.randint(0, self.nb_machine_types, self.nb_tasks)),
                 list(np.random.randint(0, self.nb_machine_types, self.nb_tasks))]
        self.machine_types = [0, 1, 0]

        machines = [(m_pos, m_typ) for m_pos, m_typ in zip(self.goal_positions, self.machine_types)]
        machines_pos_typ_0 = [machine[0] for machine in machines if machine[1] == 0]
        machines_pos_typ_1 = [machine[0] for machine in machines if machine[1] == 1]
        self.machines = [machines_pos_typ_0, machines_pos_typ_1]

        if np.sum(self.priorities) == 1:
            if self.valuation_training:
                self.priorities = [0, 1]
            else:
                self.priorities = np.random.choice([0, 1], 2, replace=False)

        self.task_positions = [(-self.field_width/2 + (1 + (i * 2)),
                                -self.field_height/2 + -1) for i in range(self.nb_tasks)]

        self.debt_balance_position = [(-self.field_width/2 + 3, self.field_height/2 + 2)]
        self.greedy = []

        self.current_suggestions = np.zeros((self.nb_agents, 1), dtype=int)
        self.missing_suggestions = np.zeros(self.nb_agents, dtype=int)
        self.trade_positions = []
        if self.trading_steps > 0:
            pos_off = 0
            for i in range(self.trading_steps * self.nb_agents):
                self.colors['trade-{}'.format(i)] = (1.0, 1.0, 1.0, 0.0)
                self.trade_positions.append((-self.field_width / 2 + 7, -self.field_height / 2 + 5 - pos_off))
                if i % 2 == 1:
                    pos_off += 2

        for i in range(self.nb_agents):
            agent = Agent(world=self.world,
                          env=self,
                          index=i,
                          position=self.possible_positions[spawning_indices[i]],
                          task=tasks[i])
            drawing_util.add_polygon(self.display_objects, agent.body, agent.agent_vertices,
                                     name='agent-{}'.format(i),
                                     drawing_layer=2,
                                     color=agent.color)
            self.agents.append(agent)
            agent.camera.pos[0] = agent.body.transform.position.x - 0.5
            agent.camera.pos[1] = agent.body.transform.position.y - 0.5

        for i, wall_pos in enumerate(self.wall_positions):
            drawing_util.add_polygon_at_pos(self.display_objects,
                                            position=wall_pos,
                                            vertices=self.agents[0].agent_vertices,
                                            name='wall-{}'.format(i),
                                            drawing_layer=2,
                                            color=self.colors['wall'])

        for i, goal_pos in enumerate(self.goal_positions):
            self.goals.append(GridCell(env=self,
                                       index=i,
                                       position=goal_pos,
                                       typ=self.machine_types[i]))
            drawing_util.add_polygon_at_pos(self.display_objects,
                                            position=(goal_pos[0], goal_pos[1]),
                                            vertices=self.agents[0].agent_vertices,
                                            name='machine-{}'.format(i),
                                            drawing_layer=0,
                                            color=self.colors['machine-{}'.format(self.goals[-1].typ)])

        for i, task_pos in enumerate(self.task_positions):
            self.tasks.append(GridCell(env=self,
                                       index=i,
                                       position=task_pos,
                                       typ='task-{}'.format(i)))
            drawing_util.add_polygon_at_pos(self.display_objects,
                                            position=(task_pos[0], task_pos[1]),
                                            vertices=self.agents[0].agent_vertices,
                                            name='task-{}'.format(i),
                                            drawing_layer=0,
                                            color=self.colors['outer_field'.format(1)])

        for i, debt_balance_pos in enumerate(self.debt_balance_position):
            self.debt_balances.append(GridCell(env=self,
                                               index=0,
                                               position=debt_balance_pos,
                                               typ='debt_balance'))
            drawing_util.add_polygon_at_pos(self.display_objects,
                                            position=(debt_balance_pos[0], debt_balance_pos[1]),
                                            vertices=self.agents[0].agent_vertices,
                                            name='debt_balance',
                                            drawing_layer=0,
                                            color=(1.0, 1.0, 1.0, 0.0))

        if self.trading_steps > 0:
            for i, trade_pos in enumerate(self.trade_positions):
                drawing_util.add_polygon_at_pos(self.display_objects,
                                                position=(trade_pos[0], trade_pos[1]),
                                                vertices=self.agents[0].agent_vertices,
                                                name='trade-{}'.format(i),
                                                drawing_layer=0,
                                                color=self.colors['trade-{}'.format(i)])

        self._create_field()

        return self.observation

    def get_state(self):
        agent_states = []
        for agent in self.agents:
            state = agent.get_state()
            agent_states.append(state)

        state = Smartfactory.State(agent_states=agent_states)

        return state

    def set_state(self, state):
        for agent, ag_state in zip(self.agents, state.agent_states):
            agent.set_state(ag_state=ag_state)

    def store_values(self):
        agent_states = []
        agent_tasks = []
        agent_done = []
        for agent in self.agents:
            agent_states.append(deepcopy(agent.get_state()))
            agent_tasks.append(deepcopy(agent.task))
            agent_done.append(deepcopy(agent.done))

        goals = []
        for g in self.goals:
            goals.append(deepcopy(g.inactive))

        if self.render_mode:
            display_obs = [self.display_objects['agent-0'][1].color, self.display_objects['agent-1'][1].color,
                       self.display_objects['machine-0'][1].color, self.display_objects['machine-1'][1].color,
                       self.display_objects['machine-2'][1].color, self.display_objects['task-0'][1].color,
                       self.display_objects['task-1'][1].color, self.display_objects['task-2'][1].color,
                       self.display_objects['debt_balance'][1].color, self.display_objects['trade-0'][1].color,
                       self.display_objects['trade-1'][1].color, self.display_objects['outer_field'][1].color,
                       self.display_objects['field'][1].color]
        else:
            display_obs = []

        sf_values = [agent_states, agent_tasks, agent_done, deepcopy(self.colors), deepcopy(self.current_suggestions),
                     goals, display_obs]

        return sf_values

    def load_values(self, sf_values):

        for i in range(self.nb_agents):
            self.agents[i].set_state(sf_values[0][i])
            self.agents[i].task = sf_values[1][i]
            self.agents[i].done = sf_values[2][i]
        self.colors = sf_values[3]
        self.current_suggestions = sf_values[4]
        for i in range(len(self.goals)):
            self.goals[i].inactive = sf_values[5][i]

        if self.render_mode:
            self.display_objects['agent-0'][1].color = sf_values[6][0]
            self.display_objects['agent-1'][1].color = sf_values[6][1]
            self.display_objects['machine-0'][1].color = sf_values[6][2]
            self.display_objects['machine-1'][1].color = sf_values[6][3]
            self.display_objects['machine-2'][1].color = sf_values[6][4]
            self.display_objects['task-0'][1].color = sf_values[6][5]
            self.display_objects['task-1'][1].color = sf_values[6][6]
            self.display_objects['task-2'][1].color = sf_values[6][7]
            self.display_objects['debt_balance'][1].color = sf_values[6][8]
            self.display_objects['trade-0'][1].color = sf_values[6][9]
            self.display_objects['trade-1'][1].color = sf_values[6][10]
            self.display_objects['outer_field'][1].color = sf_values[6][11]
            self.display_objects['field'][1].color = sf_values[6][12]

    def step(self, actions):
        """
        :param actions: the list of agent actions
        :type actions: list
        """
        info = copy.deepcopy(self.info)
        rewards = np.zeros(self.nb_agents)
        done = False

        if self.learning == decentral_learning:
            joint_actions = []
            for i_ag in range(self.nb_agents):
                joint_actions.append(self.actions[actions[i_ag]])
            actions = joint_actions

        if any([agent.done for agent in self.agents]):
            for agent in self.agents:
                agent.episode_debts = 0

        queue = np.random.choice([0, self.nb_agents-1], self.nb_agents, replace=False)
        for i in queue:
            agent = self.agents[i]
            self.set_log(i, actions[agent.index])
            if not agent.done:
                self.set_position(agent, actions[agent.index])

                if self.priorities[i]:
                    rewards[i] -= self.step_penalties[0]
                else:
                    rewards[i] -= self.step_penalties[1]

                if agent.process_task_sequential() >= 0:
                    if self.priorities[i]:
                        rewards[i] += self.rewards[0]
                    else:
                        rewards[i] += self.rewards[1]

                if agent.tasks_finished():
                    agent.done = True
                    # if self.priorities[i] and not self.agents[(i + 1) % 2].done:
                    #    rewards[i] += self.rewards[1]


        self.process_machines()

        if np.sum([int(agent.done) for agent in self.agents]) == len(self.agents):
            done = True

        return self.observation, rewards, [agent.done for agent in self.agents], info

    @property
    def observation(self):
        """
        OpenAI Gym Observation
        :return:
            List of observations
        """
        observations = []
        for i_agent, agent in enumerate(self.agents):
            if self.observation_space.shape == (84, 84, 1):
                observation = self.render(mode='rgb_array', agent_id=i_agent)
                observations.append(observation)
            elif self.observation_space.shape == (9, self.field_height, self.field_height):
                observation = self.observation_one_hot(i_agent)
                observations.append(observation)
            elif self.observation_space.shape == (11, self.field_height, self.field_height):
                observation = self.observation_trade(i_agent)
                observations.append(observation)
            elif self.observation_space.shape == (12, self.field_height, self.field_height):
                observation = self.observation_trade_suggestion(i_agent)
                observations.append(observation)

        return observations

    def get_observation_trade(self, agent_index):
        observation = []
        if self.observation_space.shape == (84, 84, 1):
            observation = self.render(mode='rgb_array', agent_id=agent_index)
        elif self.observation_space.shape == (9, self.field_height, self.field_height):
            observation = self.observation_one_hot(agent_index)
        elif self.observation_space.shape == (11, self.field_height, self.field_height):
            observation = self.observation_trade(agent_index)
        elif self.observation_space.shape == (12, self.field_height, self.field_height):
            observation = self.observation_trade_suggestion(agent_index)
        return observation

    def get_valuation_observation(self, agent_index):
        observation = self.observation_one_hot(agent_index)
        return observation

    def set_render_mode(self, render_mode):
        self.render_mode = render_mode

    def set_valuation_training(self, valuation_training):
        self.valuation_training = valuation_training

    def set_position(self, agent, action):

        agent.old_pos = agent.current_pos

        new_pos = [agent.body.transform.position.x + (1.0 * action[0]),
                   agent.body.transform.position.y + (1.0 * action[1])]

        if new_pos not in self.wall_positions:
            if action[1] == 1.0: # up
                if new_pos[1] <= self.field_vertices[2][1]:
                    agent.body.transform = (new_pos, 0)
                    agent.current_pos = new_pos
            if action[1] == -1.0: # down
                if new_pos[1] > self.field_vertices[0][1]:
                    agent.body.transform = (new_pos, 0)
                    agent.current_pos = new_pos
            if action[0] == -1.0: # left
                if new_pos[0] > self.field_vertices[0][0]:
                    agent.body.transform = (new_pos, 0)
                    agent.current_pos = new_pos
            if action[0] == 1.0: # right
                if new_pos[0] <= self.field_vertices[1][0]:
                    agent.body.transform = (new_pos,0)
                    agent.current_pos = new_pos

        agent.camera.pos[0] = agent.body.transform.position.x - 0.5
        agent.camera.pos[1] = agent.body.transform.position.y - 0.5

    def process_machines(self):
        self.display_objects['field'][1].color = self.colors['field']
        for i_machine, machine in enumerate(self.goals):
            machine_index = 'machine-{}'.format(i_machine)
            if machine.inactive <= 0:
                machine_type = 'machine-{}'.format(machine.typ)
                self.display_objects[machine_index][1].color = self.colors[machine_type]
            else:
                machine.inactive -= 1
                self.display_objects[machine_index][1].color = (1.0, 1.0, 1.0, 0.0)

    def check_contracting(self, actions):

        greedy = [0, 0]
        contracting = False

        if self.contracting == 0:
            pass

        if self.contracting == 1:
            raise NotImplementedError

        if self.contracting == 2:
            if self.actions[actions[0]][2] == 1 and self.actions[actions[1]][3] == 1:
                contracting = True
                greedy[1] = 1
            if self.actions[actions[0]][3] == 1 and self.actions[actions[1]][2] == 1:
                contracting = True
                greedy[0] = 1

        assert any(i == 0 for i in greedy)
        self.greedy = greedy

        return contracting, greedy

    def set_log(self, agent_index, action):
        taken_action = [agent_index, action]
        self.actions_log.append(taken_action)

    def get_current_actions(self):
        current_actions = []
        for i_actions in range(self.nb_agents):
            current_actions.append(self.actions_log.pop(0))
        return current_actions

    def update_trade_colors(self, suggested_steps):
        self.current_suggestions = suggested_steps

        if self.render_mode:
            suggested_steps_copy = deepcopy(self.current_suggestions)
            for i in range(len(suggested_steps)):
                for i_steps in range(self.trading_steps):
                    if suggested_steps_copy[i]:
                        if suggested_steps_copy[i][1] == 1.0:  # up
                            self.colors['trade-{}'.format(i + i_steps * 2)] = (0.0, 0.0, 0.0, 1.0)
                        if suggested_steps_copy[i][1] == -1.0:  # down
                            self.colors['trade-{}'.format(i + i_steps * 2)] = (0.33, 0.33, 0.33, 1.0)
                        if suggested_steps_copy[i][0] == -1.0:  # left
                            self.colors['trade-{}'.format(i + i_steps * 2)] = (0.66, 0.66, 0.66, 1.0)
                        if suggested_steps_copy[i][0] == 1.0:  # right
                            self.colors['trade-{}'.format(i + i_steps * 2)] = (1.0, 1.0, 1.0, 1.0)
                        suggested_steps_copy[i] = suggested_steps_copy[i][2:]
                    else:
                        self.colors['trade-{}'.format(i + i_steps * 2)] = (1.0, 1.0, 1.0, 0.0)

        return self.observation

    def set_suggestions(self, suggested_steps):
        self.current_suggestions = suggested_steps

    def set_missing_suggestion(self, agent_index, nb_suggestion):
        self.missing_suggestions[agent_index] = nb_suggestion

    def render(self, mode='human', close=False, info_values=None, agent_id=None, video=False):
        if mode == 'rgb_array':
            display_objects = self.display_objects.copy()
            if video:
                camera = Camera(pos=(0,0), fov_dims=(9, 9))
            else:
                camera = self.camera

            if agent_id is not None:

                if np.sum([int(agent.done) for agent in self.agents]) > 0:
                    display_objects['field'][1].color = self.colors['dark']
                else:
                    display_objects['field'][1].color = self.colors['white']

                if self.agents[agent_id].done:
                    display_objects['agent-{}'.format(agent_id)][1].color = (1.0, 1.0, 1.0, 0.0)
                else:
                    display_objects['agent-{}'.format(agent_id)][1].color = self.colors['agent-0']

                display_objects['agent-{}'.format(agent_id)] = (10, display_objects['agent-{}'.format(agent_id)][1])

                if self.priorities[agent_id]:
                    display_objects['debt_balance'][1].color = self.colors['white']
                else:
                    display_objects['debt_balance'][1].color = (1.0, 1.0, 1.0, 0.0)

                for t in range(self.nb_tasks):
                    display_objects['task-{}'.format(t)][1].color = (1.0, 1.0, 1.0, 0.0)

                    if len(self.agents[agent_id].task) > t:
                        task = self.agents[agent_id].task[t]

                        if task >= 0:
                            display_objects['task-{}'.format(t)][1].color = self.colors['machine-{}'.format(task)]
                    # else:

                if self.trading_steps > 0:
                    for i in range(self.trading_steps * self.nb_agents):
                        display_objects['trade-{}'.format(i)][1].color = (1.0, 1.0, 1.0, 0.0)
                        if i % 2 == agent_id:
                            display_objects['trade-{}'.format(i)][1].color = self.colors['trade-{}'.format(i)]

                #if np.sum(self.greedy) > 0:
                #    i = self.greedy.index(0)
                #    if i == agent_id:
                #            display_objects['field'][1].color = self.colors['dark']

                for i_agent, agent in enumerate(self.agents):
                    if i_agent != agent_id:
                        if self.agents[i_agent].done:
                            display_objects['agent-{}'.format(i_agent)][1].color = (1.0, 1.0, 1.0, 0.0)
                        else:
                            display_objects['agent-{}'.format(i_agent)][1].color = self.colors['agent-1']
                        display_objects['agent-{}'.format((i_agent))] = (2, display_objects['agent-{}'.format(i_agent)][1])

            return render_visual_state({'camera': camera,
                                        'display_objects': display_objects},
                                       info_values,
                                       pixels_per_worldunit=self.pixels_per_worldunit)

    def observation_one_hot(self, agent_id):

        channels = 9
        observation = np.zeros((channels, self.field_width, self.field_height))

        c_active_machines = 0
        c_pos_self = 1
        c_task_prio = 2
        c_tasks = [3, 4, 5]

        c_pos_other = 6
        c_task_0_other = 7
        c_done_other = 8

        for g in self.goals:
            x_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][0])
            y_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][1])

            if g.inactive <= 0:
                observation[c_active_machines][x_m][y_m] += 1

        x_a_raw = self.agents[agent_id].body.transform.position.x
        y_a_raw = self.agents[agent_id].body.transform.position.y
        x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
        y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
        observation[c_pos_self][x_a][y_a] += 1

        for i_task, task in enumerate(self.agents[agent_id].task):
            for x_task_raw, y_task_raw in self.machines[task]:
                x_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][0])
                y_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][1])

                observation[c_tasks[i_task]][x_task][y_task] += 1

        # other agent
        if len(self.agents) > 1:
            x_a_raw = self.agents[(agent_id + 1) % 2].body.transform.position.x
            y_a_raw = self.agents[(agent_id + 1) % 2].body.transform.position.y
            x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
            y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
            observation[c_pos_other][x_a][y_a] += 1

            if len(self.agents[(agent_id + 1) % 2].task) > 0:
                for x_task_other_raw, y_task_other_raw in self.machines[self.agents[(agent_id + 1) % 2].task[0]]:
                    x_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][0])
                    y_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][1])
                    observation[c_task_0_other][x_task_other][y_task_other] += 1

        observation[c_task_prio] += self.priorities[agent_id]

        if self.agents[(agent_id + 1) % 2].done:
            observation[c_done_other] += 1

        return observation

    def observation_trade(self, agent_id):

        channels = 11
        observation = np.zeros((channels, self.field_width, self.field_height))

        c_active_machines = 0
        c_pos_self = 1
        c_task_prio = 2
        c_tasks = [3, 4, 5]

        c_pos_other = 6
        c_task_0_other = 7
        c_done_other = 8

        c_suggestions = 9
        c_suggestions_other = 10

        for g in self.goals:
            x_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][0])
            y_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][1])

            if g.inactive <= 0:
                observation[c_active_machines][x_m][y_m] += 1

        x_a_raw = self.agents[agent_id].body.transform.position.x
        y_a_raw = self.agents[agent_id].body.transform.position.y
        x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
        y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
        observation[c_pos_self][x_a][y_a] += 1

        for i_task, task in enumerate(self.agents[agent_id].task):
            for x_task_raw, y_task_raw in self.machines[task]:
                x_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][0])
                y_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][1])

                observation[c_tasks[i_task]][x_task][y_task] += 1

        # other agent
        if len(self.agents) > 1:
            x_a_raw = self.agents[(agent_id + 1) % 2].body.transform.position.x
            y_a_raw = self.agents[(agent_id + 1) % 2].body.transform.position.y
            x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
            y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
            observation[c_pos_other][x_a][y_a] += 1

            if len(self.agents[(agent_id + 1) % 2].task) > 0:
                for x_task_other_raw, y_task_other_raw in self.machines[self.agents[(agent_id + 1) % 2].task[0]]:
                    x_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][0])
                    y_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][1])
                    observation[c_task_0_other][x_task_other][y_task_other] += 1

        observation[c_task_prio] += self.priorities[agent_id]

        x_pos = self.agents[agent_id].body.transform.position.x
        y_pos = self.agents[agent_id].body.transform.position.y
        x_off = 0
        y_off = 0

        for i_steps in range(int(len(self.current_suggestions[agent_id])/2)):
            x_step = int(self.current_suggestions[agent_id][i_steps * 2])
            y_step = int(self.current_suggestions[agent_id][i_steps * 2 + 1])

            x_sugg = int(self.map['{}-{}'.format(x_pos, y_pos)][0]) + x_step + x_off
            y_sugg = int(self.map['{}-{}'.format(x_pos, y_pos)][1]) + y_step + y_off

            if 0 <= x_sugg < self.field_width and 0 <= y_sugg < self.field_height:
                observation[c_suggestions][x_sugg][y_sugg] += 1
                x_off += x_step
                y_off += y_step
            else:
                observation[c_suggestions][x_sugg - x_step][y_sugg - y_step] += 1

        x_pos_o = self.agents[(agent_id + 1) % 2].body.transform.position.x
        y_pos_o = self.agents[(agent_id + 1) % 2].body.transform.position.y
        x_off = 0
        y_off = 0

        for i_steps in range(int(len(self.current_suggestions[(agent_id + 1) % 2]) / 2)):
            x_step = int(self.current_suggestions[(agent_id + 1) % 2][i_steps * 2])
            y_step = int(self.current_suggestions[(agent_id + 1) % 2][i_steps * 2 + 1])

            x_sugg_o = int(self.map['{}-{}'.format(x_pos_o, y_pos_o)][0]) + x_step + x_off
            y_sugg_o = int(self.map['{}-{}'.format(x_pos_o, y_pos_o)][1]) + y_step + y_off

            if 0 <= x_sugg_o < self.field_width and 0 <= y_sugg_o < self.field_height:
                observation[c_suggestions_other][x_sugg_o][y_sugg_o] += 1
                x_off += x_step
                y_off += y_step
            else:
                observation[c_suggestions_other][x_sugg_o - x_step][y_sugg_o - y_step] += 1

        if self.agents[(agent_id + 1) % 2].done:
            observation[c_done_other] += 1

        return observation

    def observation_trade_suggestion(self, agent_id):

        channels = 12
        observation = np.zeros((channels, self.field_width, self.field_height))

        c_active_machines = 0
        c_pos_self = 1
        c_task_prio = 2
        c_tasks = [3, 4, 5]

        c_action_mode = 6

        c_pos_other = 7
        c_task_0_other = 8
        c_done_other = 9

        c_suggestions = 10
        c_suggestions_other = 11

        for g in self.goals:
            x_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][0])
            y_m = int(self.map['{}-{}'.format(g.position[0], g.position[1])][1])

            if g.inactive <= 0:
                observation[c_active_machines][x_m][y_m] += 1

        x_a_raw = self.agents[agent_id].body.transform.position.x
        y_a_raw = self.agents[agent_id].body.transform.position.y
        x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
        y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
        observation[c_pos_self][x_a][y_a] += 1

        for i_task, task in enumerate(self.agents[agent_id].task):
            for x_task_raw, y_task_raw in self.machines[task]:
                x_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][0])
                y_task = int(self.map['{}-{}'.format(x_task_raw, y_task_raw)][1])

                observation[c_tasks[i_task]][x_task][y_task] += 1

        # other agent
        if len(self.agents) > 1:
            x_a_raw = self.agents[(agent_id + 1) % 2].body.transform.position.x
            y_a_raw = self.agents[(agent_id + 1) % 2].body.transform.position.y
            x_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][0])
            y_a = int(self.map['{}-{}'.format(x_a_raw, y_a_raw)][1])
            observation[c_pos_other][x_a][y_a] += 1

            if len(self.agents[(agent_id + 1) % 2].task) > 0:
                for x_task_other_raw, y_task_other_raw in self.machines[self.agents[(agent_id + 1) % 2].task[0]]:
                    x_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][0])
                    y_task_other = int(self.map['{}-{}'.format(x_task_other_raw, y_task_other_raw)][1])
                    observation[c_task_0_other][x_task_other][y_task_other] += 1

        observation[c_task_prio] += self.priorities[agent_id]

        x_pos = self.agents[agent_id].body.transform.position.x
        y_pos = self.agents[agent_id].body.transform.position.y
        x_off = 0
        y_off = 0

        for i_steps in range(int(len(self.current_suggestions[agent_id])/2)):
            x_step = int(self.current_suggestions[agent_id][i_steps * 2])
            y_step = int(self.current_suggestions[agent_id][i_steps * 2 + 1])

            x_sugg = int(self.map['{}-{}'.format(x_pos, y_pos)][0]) + x_step + x_off
            y_sugg = int(self.map['{}-{}'.format(x_pos, y_pos)][1]) + y_step + y_off

            if 0 <= x_sugg < self.field_width and 0 <= y_sugg < self.field_height:
                observation[c_suggestions][x_sugg][y_sugg] += 1
                x_off += x_step
                y_off += y_step
            else:
                observation[c_suggestions][x_sugg - x_step][y_sugg - y_step] += 1

        x_pos_o = self.agents[(agent_id + 1) % 2].body.transform.position.x
        y_pos_o = self.agents[(agent_id + 1) % 2].body.transform.position.y
        x_off = 0
        y_off = 0

        for i_steps in range(int(len(self.current_suggestions[(agent_id + 1) % 2]) / 2)):
            x_step = int(self.current_suggestions[(agent_id + 1) % 2][i_steps * 2])
            y_step = int(self.current_suggestions[(agent_id + 1) % 2][i_steps * 2 + 1])

            x_sugg_o = int(self.map['{}-{}'.format(x_pos_o, y_pos_o)][0]) + x_step + x_off
            y_sugg_o = int(self.map['{}-{}'.format(x_pos_o, y_pos_o)][1]) + y_step + y_off

            if 0 <= x_sugg_o < self.field_width and 0 <= y_sugg_o < self.field_height:
                observation[c_suggestions_other][x_sugg_o][y_sugg_o] += 1
                x_off += x_step
                y_off += y_step
            else:
                observation[c_suggestions_other][x_sugg_o - x_step][y_sugg_o - y_step] += 1

        if self.agents[(agent_id + 1) % 2].done:
            observation[c_done_other] += 1

        if self.missing_suggestions[agent_id] > 0:
            observation[c_action_mode] += 1
        # observation[c_action_mode] += self.missing_suggestions[agent_id]

        return observation

    @staticmethod
    def overlaps_checkpoint(checkpoint, agent):
        return b2TestOverlap(
            checkpoint.shape, 0,
            agent.body.fixtures[0].shape, 0, b2Transform(checkpoint.position, b2Rot(0.0)),
            agent.body.transform)

    @staticmethod
    def distance_agents(agent_1, agent_2):
        dist_x = agent_1.body.transform.position.x - agent_2.body.transform.position.x
        dist_y = agent_1.body.transform.position.y - agent_2.body.transform.position.y
        return np.sqrt((dist_x*dist_x) + (dist_y*dist_y))

    @staticmethod
    def agent_collision(agent_1, agent_2):
        collision = False
        if agent_1.old_pos == agent_2.current_pos and agent_1.current_pos == agent_2.old_pos:
            collision = True
        if agent_1.current_pos == agent_2.current_pos:
            collision = True
        return collision


def make_smart_factory(params):
    # normal exploding action space
    if params.trading_mode == 0:
        action_space = trading.setup_action_space(params.trading_steps, params.trading_steps, None)
    # split action and suggestion
    else:
        # action_space = [[0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]

        action_space = [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]

    # normal observation with exploding action space
    observation = 2
    # observation with suggestion action indicator
    if params.trading_mode == 2:
        observation = 3
    # observation to train valuation nets
    if params.eval_mode < 0:
        observation = 1

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       trading_steps=params.trading_steps,
                       trading_actions=action_space,
                       priorities=params.priorities,
                       nb_machine_types=params.nb_machine_types,
                       nb_steps_machine_inactive=params.nb_steps_machine_inactive,
                       nb_tasks=params.nb_tasks,
                       observation=observation)
    return env


def make_plot(params, log_dir, exp_time):
    plt.figure(figsize=(16, 9))
    df = pd.read_csv(os.path.join(log_dir, 'values {}.csv'.format(exp_time)))

    hue_str = ""
    if params.eval_mode == 0:
        hue_str = "trading_steps"
    if params.eval_mode == 1:
        hue_str = "trading_budget"
    if params.eval_mode == 2:
        hue_str = "mark_up"

    plot = sns.boxplot(x="agent", y="reward", hue=hue_str, data=df, palette="Set1", showmeans=True)
    plot.set(ylim=(-25, 10))
    fig = plot.get_figure()
    fig.savefig(os.path.join(log_dir, 'hist {}.png'.format(exp_time)))
    # plt.show()


def main():

    with open(os.path.join('..', 'params.json'), 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    episodes = 10
    episode_steps = 500

    eval_date = '20200119-22-31-47'

    mode_str, eval_list = trading.eval_mode_setup(params)

    log_dir = os.path.join('..', 'exp-trading/{} - tr mode {} - {}/'.format(eval_date, params.trading_mode, mode_str))
    columns = ['trading_steps', 'episode', 'reward', 'accumulated_transfer', 'number_trades', 'mark_up', 'trading_budget', 'episode_steps', 'agent']

    if params.partial_pay:
        params.eval_mode = 0
        eval_list = [10]
        for i_trading_step in range(params.trading_steps+1):
            columns.append('partial {}'.format(i_trading_step))

    df = pd.DataFrame(columns=columns)

    for i_values in eval_list:
        if params.eval_mode == 0:
            params.trading_steps = i_values
        if params.eval_mode == 1:
            params.trading_budget[0] = i_values
            params.trading_budget[1] = i_values
        if params.eval_mode == 2:
            params.mark_up = i_values

        env = make_smart_factory(params)
        observation_shape = list(env.observation_space.shape)
        number_of_actions = env.action_space.n

        agents = []
        suggestion_agents = []

        for i_ag in range(params.nb_agents):
            ag = make_dqn_agent(params, observation_shape, number_of_actions)
            ag.load_weights(os.path.join(log_dir, "{} {}/weights-{}.pth".format(mode_str, i_values, i_ag)))
            ag.epsilon = 0.01
            agents.append(ag)

        if params.trading_mode == 1:
            for i_ag in range(params.nb_agents):
                suggestion_ag = make_dqn_agent(params, observation_shape, 4)
                suggestion_ag.load_weights(os.path.join(log_dir, "{} {}/weights-sugg-{}.pth".format(mode_str, i_values, i_ag)))
                suggestion_ag.epsilon = 0.01
                suggestion_agents.append(suggestion_ag)

        if params.trading_mode == 2:
            suggestion_agents = agents

        trade = trading.Trade(env=env, params=params, agents=agents, suggestion_agents=suggestion_agents)
        done_mode = params.done_mode

        for i_episode in range(episodes):
            observations = env.reset()
            episode_rewards = np.zeros(len(env.agents))
            trade.trading_budget = deepcopy(params.trading_budget)
            trade_count = np.zeros(len(agents))
            accumulated_transfer = np.zeros(len(agents))
            joint_done = [False, False]
            taken_steps = 0

            for i_step in range(episode_steps):
                actions = []
                for agent_index in [0, 1]:
                    if not joint_done[agent_index]:
                        action = agents[agent_index].policy(observations[agent_index])
                    else:
                        action = np.random.randint(0, 4)
                    actions.append(action)

                joint_reward, next_observations, joint_done, new_trades, act_transfer = trade.trading_step(episode_rewards, env, actions)

                observations = next_observations

                taken_steps += 1

                for i in range(len(agents)):
                    episode_rewards[i] += joint_reward[i]
                    trade_count[i] += new_trades[i]
                    accumulated_transfer[i] += act_transfer[i]

                if not done_mode:
                    if all(done is True for done in joint_done) or i_step == episode_steps:
                        break
                else:
                    if joint_done.__contains__(True) or i_step == episode_steps:
                        break

            print(mode_str + ": " + str(i_values)
                  + "\t|\tEpisode: " + str(i_episode)
                  + "\t\tSteps: " + str(taken_steps)
                  + "\t\tTrades: " + str(int(np.sum(trade_count)))
                  + "\t\tRewards: " + str(np.sum(episode_rewards)))

            ep_stats = [params.trading_steps, i_episode, np.sum(episode_rewards), np.sum(accumulated_transfer), np.sum(trade_count), params.mark_up, params.trading_budget, taken_steps,
                        'overall']
            ep_stats_a1 = [params.trading_steps, i_episode, episode_rewards[0], accumulated_transfer[0], int(trade_count[0]), params.mark_up, params.trading_budget, taken_steps,
                           'a-{}'.format(1)]
            ep_stats_a2 = [params.trading_steps, i_episode, episode_rewards[1], accumulated_transfer[1], int(trade_count[1]), params.mark_up, params.trading_budget, taken_steps,
                           'a-{}'.format(2)]

            if params.partial_pay:
                for i_trading_step in range(params.trading_steps+1):
                    ep_stats.append(trade.trades[0][i_trading_step] + trade.trades[1][i_trading_step])
                    ep_stats_a1.append(trade.trades[0][i_trading_step])
                    ep_stats_a2.append(trade.trades[1][i_trading_step])

            df_ep = pd.DataFrame([ep_stats, ep_stats_a1, ep_stats_a2], columns=columns)
            df = df.append(df_ep, ignore_index=True)

    log_dir_eval = os.path.join(log_dir, 'evaluation files')
    if not os.path.exists(log_dir_eval):
        os.makedirs(log_dir_eval)
    exp_time = datetime.now().strftime('%Y%m%d-%H-%M-%S')
    df.to_csv(os.path.join(log_dir_eval, 'values {}.csv'.format(exp_time)))

    make_plot(params, log_dir_eval, exp_time)

    print("finished")


if __name__ == '__main__':
    main()
