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
import agents.pytorch_agents as pta


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
                 trading=0,
                 trading_steps=0,
                 trading_actions=None,
                 trading_signals=None,
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
            self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(8, field_width, field_height))

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

        with open('envs/actions.json', 'r') as f:
            actions_json = json.load(f)

        self.actions = []
        self.contracting = contracting
        # if contracting == 0:
        #     self.actions = actions_json['no_contracting_action']
        # if contracting == 1:
        #     self.actions = actions_json['one_contracting_action']
        # if contracting == 2:
        #     self.actions = actions_json['two_contracting_actions']

        self.trading = trading
        self.trading_steps = trading_steps
        self.trading_signals = trading_signals

        self.actions = trading_actions

        self.actions_log = []
        self.trade_positions = []

        self.learning = learning
        self.nb_actions = len(self.actions)
        self.action_space =  gym.spaces.Discrete(self.nb_actions)
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
            self.priorities = np.random.choice([0, 1], 2, replace=False)

        self.task_positions = [(-self.field_width/2 + (1 + (i * 2)),
                                -self.field_height/2 + -1) for i in range(self.nb_tasks)]

        self.debt_balance_position = [(-self.field_width/2 + 3, self.field_height/2 + 2)]
        self.greedy = []

        self.trade_positions = []
        if self.trading != 0 and self.trading_steps != 0:
            if self.trading_signals:
                pos_off = 0
                for i in range(self.trading_steps * self.nb_agents):
                    self.colors['trade-{}'.format(i)] = (1.0, 1.0, 1.0, 0.0)
                    self.trade_positions.append((-self.field_width / 2 + 7, -self.field_height / 2 + 5 - pos_off))
                    if i % 2 == 1:
                        pos_off += 2
            else:
                for i_agents in range(self.nb_agents):
                    self.colors['trade-{}'.format(i_agents)] = (1.0, 1.0, 1.0, 0.0)
                    self.trade_positions.append((-self.field_width / 2 + 7, -self.field_height / 2 + 5))

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

        if self.trading != 0 and self.trading_steps != 0:
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
            elif self.observation_space.shape == (8, self.field_height, self.field_height):
                observation = self.observation_one_hot(i_agent)
                observations.append(observation)

        return observations

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
        observations = deepcopy(self.observation)
        if self.trading_signals:
            suggested_steps_copy = deepcopy(suggested_steps)

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
        else:
            for i in range(len(suggested_steps)):
                if suggested_steps[i]:
                    if suggested_steps[i][1] == 1.0:  # up
                        self.colors['trade-{}'.format(i)] = (0.0, 0.0, 0.0, 1.0)
                    if suggested_steps[i][1] == -1.0:  # down
                        self.colors['trade-{}'.format(i)] = (0.33, 0.33, 0.33, 1.0)
                    if suggested_steps[i][0] == -1.0:  # left
                        self.colors['trade-{}'.format(i)] = (0.66, 0.66, 0.66, 1.0)
                    if suggested_steps[i][0] == 1.0:  # right
                        self.colors['trade-{}'.format(i)] = (1.0, 1.0, 1.0, 1.0)
                else:
                    self.colors['trade-{}'.format(i)] = (1.0, 1.0, 1.0, 0.0)
        return observations

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

                if self.trading_steps > 0 and self.trading > 0:
                    render_trade = True
                    for i in range(self.nb_agents):
                        if self.agents[i].done:
                            render_trade = False

                    if self.trading_signals:
                        for i in range(self.trading_steps * self.nb_agents):
                            display_objects['trade-{}'.format(i)][1].color = (1.0, 1.0, 1.0, 0.0)
                            if render_trade:
                                if i % 2 == agent_id:
                                    display_objects['trade-{}'.format(i)][1].color = self.colors['trade-{}'.format(i)]
                    else:
                        for i_agents in range(self.nb_agents):
                            display_objects['trade-{}'.format(i_agents)][1].color = (1.0, 1.0, 1.0, 0.0)
                            if render_trade:
                                if i_agents % 2 == agent_id:
                                    display_objects['trade-{}'.format(i_agents)][1].color = self.colors[
                                        'trade-{}'.format(i_agents)]

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

        channels = 8
        observation = np.zeros((channels, self.field_width, self.field_height))

        c_active_machines = 0
        c_pos_self = 1
        c_task_prio = 2
        c_tasks = [3, 4, 5]

        c_pos_other = 6
        c_task_0_other = 7

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


def main():

    with open(os.path.join('..', 'params.json'), 'r') as f:
        params_json = json.load(f)
    params = DotMap(params_json)

    policy_random = False
    episodes = 10
    episode_steps = 100

    env = Smartfactory(nb_agents=params.nb_agents,
                       field_width=params.field_width,
                       field_height=params.field_height,
                       rewards=params.rewards,
                       step_penalties=params.step_penalties,
                       priorities=params.priorities,
                       contracting=2,
                       nb_machine_types=params.nb_machine_types,
                       nb_tasks=params.nb_tasks,
                       observation=1
                       )

    observation_shape = list(env.observation_space.shape)
    number_of_actions = env.action_space.n

    agents = []
    for i_ag in range(params.nb_agents):
        ag = pta.DqnAgent(
            observation_shape=observation_shape,
            number_of_actions=number_of_actions,
            gamma=0.95,
            epsilon_decay=0.00002,
            epsilon_min=0.0,
            mini_batch_size=64,
            warm_up_duration=1000,
            buffer_capacity=20000,
            target_update_period=2000,
            seed=1337)
        ag.epsilon = 0.01
        agents.append(ag)

    for i_agent, agent in enumerate(agents):
        agent.load_weights('/Users/kyrill/Documents/research/contracting-agents/weights.{}.pth'.format(i_agent))

    episodes = 10
    episode_steps = 100
    combined_frames = []
    for i_episode in range(episodes):

        observations = env.reset()
        episode_rewards = np.zeros(params.nb_agents)
        episode_contracts = 0
        combined_frames = drawing_util.render_combined_frames(combined_frames, env, [0, 0], 0, [0, 0])

        for i_step in range(episode_steps):
            actions = []
            for i_ag, agent in enumerate(agents):
                if not env.agents[i_ag].done:
                    action = agents[i_ag].policy(observations[i_ag])
                    actions.append(action)
                else:
                    actions.append(0)

            observations, r, done, info = env.step(actions)
            episode_rewards += r

            combined_frames = drawing_util.render_combined_frames(combined_frames, env, r, 0, actions)

            if all([agent.done for agent in env.agents]):
                break

        print("Episode {} contracts: {}".format(i_episode, episode_contracts))
        print(episode_rewards)
    export_video('Smart-Factory-Contracting.mp4', combined_frames, None)

if __name__ == '__main__':
    main()
