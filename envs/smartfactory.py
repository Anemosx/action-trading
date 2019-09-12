from common_utils.utils import export_video
from rl.core import Processor

import gym
import gym.spaces
from collections import namedtuple
import common_utils.drawing_util as drawing_util
from common_utils.drawing_util import Camera, render_visual_state
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

INPUT_SHAPE = (84, 84)
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

        self.camera = Camera(pos=deepcopy(position), fov_dims=(3, 3))
        self.signalling = False
        self.task = task
        self.done = False

    def process_task(self):
        x = self.body.transform.position.x
        y = self.body.transform.position.y

        task_index = -1
        if [x, y] in self.env.goal_positions:
            index_machine = self.env.goal_positions.index([x,y])
            machine = self.env.goals[index_machine]
            if machine.typ in self.task and machine.inactive <= 0:
                machine.inactive = 10
                index_task = self.task.index(machine.typ)
                self.task[index_task] = -1
                task_index = index_task
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
            img = img.resize(INPUT_SHAPE)  # .convert('L')  # resize and convert to grayscale
            processed_observation = np.array(img)
            # assert processed_observation.shape == INPUT_SHAPE
            return processed_observation.astype('uint8')  # saves storage in experience memory

        def process_state_batch(self, batch):
            # We could perform this processing step in `process_observation`. In this case, however,
            # we would need to store a `float32` array instead, which is 4x more memory intensive than
            # an `uint8` array. This matters if we store 1M observations.

            processed_batch = batch.astype('float32') / 255.
            # processed_batch = processed_batch.reshape(len(batch), 84, 84, 3)
            processed_batch = processed_batch.reshape(len(batch), 84, 84, 3)
            return processed_batch

        def process_reward(self, reward):
            return np.clip(reward, -1., 5.)

    def __init__(self,
                 nb_agents,
                 field_width,
                 field_height,
                 rewards,
                 learning=decentral_learning,
                 contracting=2,
                 nb_machine_types=2,
                 nb_tasks=3):
        """

        :rtype: observation
        """
        self.world = Box2D.b2World(gravity=(0, 0))
        self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(84, 84, 3))
        self.velocity_iterations = 6
        self.position_iterations = 2
        self.dt = 1.0 / 15
        self.agent_restitution = 0.5
        self.agent_density = 1.0

        self.colors = {
            'agent-0': (0.20392156862745098, 0.596078431372549, 0.8588235294117647),
            'agent-1': (0.9058823529411765, 0.2980392156862745, 0.23529411764705882),
            'outer_field': (0.5843137254901961, 0.6470588235294118, 0.6509803921568628),
            'field': (1.0, 1.0, 1.0, 1.0),
            'wall': (0.5843137254901961, 0.6470588235294118, 0.6509803921568628, 0.4),
            'checkpoint': (0.13, 0.15, 0.14, 1.0),
            'machine-0': (0.6078431372549019, 0.34901960784313724, 0.7137254901960784),
            'machine-1': (0.1803921568627451, 0.8, 0.44313725490196076),
            'contracting': (0.13, 0.15, 0.14, 1.0),
        }

        with open('/Users/kyrill/Documents/research/wicksellian-triangle/envs/actions.json','r') as f:
            actions_json = json.load(f)

        self.actions = []
        if contracting == 0:
            self.actions = actions_json['no_contracting_action']
            self.colors['field'] = (0.13, 0.15, 0.14, 1.0)
        if contracting == 1:
            self.actions = actions_json['one_contracting_action']
            self.colors['field'] = (1.0, 1.0, 1.0, 1.0)
        if contracting == 2:
            self.actions = actions_json['two_contracting_actions']
            self.colors['field'] = (1.0, 1.0, 1.0, 1.0)

        self.learning = learning
        if learning == joint_learning:
            self.actions = list(it.product(self.actions, self.actions))
        self.nb_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(n=len(self.actions))

        self.info = {'setting': None,
                     'episode': None,
                     'return_a0': None,
                     'return_a1': None
                     }

        self.pixels_per_worldunit = 24
        self.obs_pixels_per_worldunit = 8
        self.camera = Camera(pos=(0, 0), fov_dims=(9, 9))
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
        self.contract = False

        self.nb_machine_types = nb_machine_types
        self.nb_tasks = nb_tasks

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
        self.display_objects = dict()
        self.agents = []
        self.market_agents = []
        self.food = []
        self.goals = []
        self.tasks = []
        self.contract = False
        self.possible_positions = [[(-(self.field_width-(self.field_width/2)-1))+column,
                                    (self.field_height/2)-row] for row in range(self.field_height)
                                   for column in range(self.field_width)]

        field_indices = [pos for pos in range(self.field_width*self.field_height)]
        wall_indices = []
        goal_indices = [0, 8, 24]
        self.wall_positions = [self.possible_positions[i] for i in wall_indices]
        self.goal_positions = [self.possible_positions[i] for i in goal_indices]
        spawning_positions = list(set(field_indices) - set(wall_indices) - set(goal_indices))
        spawning_indices = np.random.choice(spawning_positions, self.nb_agents, replace=False)

        tasks = [list(np.random.randint(0, self.nb_machine_types, self.nb_tasks)),
                 list(np.random.randint(0, self.nb_machine_types, self.nb_tasks))]
        machine_types = [0, 1, 0, 1, 0]

        # spawning_indices[0] = 72
        # spawning_indices[1] = 73

        self.task_positions = [(-self.field_width/2 + (1 + (i * 2)) , -self.field_height/2 + -1) for i in range(self.nb_tasks)]

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
                                       typ=machine_types[i]))
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
        self.display_objects['field'][1].color = self.colors['field']
        if self.contract:
            self.display_objects['field'][1].color = self.colors['contracting']


        info = copy.deepcopy(self.info)
        rewards = np.zeros(self.nb_agents)

        if self.learning == decentral_learning:
            joint_actions = [self.actions[actions[0]], self.actions[actions[1]]]
            actions = joint_actions

        if self.learning == joint_learning:
            joint_actions = [self.actions[actions[0]][0], self.actions[actions[0]][1]]
            actions = joint_actions

        done = False
        for i, agent in enumerate(self.agents):
            self.set_position(agent, actions[agent.index])
            agent.process_task()
            if agent.tasks_finished():
                rewards[i] += self.rewards[i]
                done = True
        rewards -= 0.004

        for i_machine, machine in enumerate(self.goals):
            if machine.inactive <= 0:
                self.display_objects['machine-{}'.format(i_machine)][1].color = self.colors['machine-{}'.format(machine.typ)]
            else:
                machine.inactive -= 1
                self.display_objects['machine-{}'.format(i_machine)][1].color = self.colors['field']


        for i in range(self.nb_agents):
            info['return_a{}'.format(i)] = rewards[i]

        if self.learning == joint_learning:
            rewards = [np.sum(rewards)]

        return self.observation, rewards, done, info

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

    def render(self, mode='human', close=False, info_values=None, agent_id=None):
        if mode == 'rgb_array':
            display_objects = self.display_objects.copy()
            if agent_id is not None:
                for i_agent, agent in enumerate(self.agents):
                    if i_agent == agent_id:
                        display_objects['agent-{}'.format(i_agent)][1].color = self.colors['agent-0']
                        for i_task, task in enumerate(agent.task):
                            if task >= 0:
                                display_objects['task-{}'.format(i_task)][1].color = self.colors['machine-{}'.format(task)]
                            else:
                                display_objects['task-{}'.format(i_task)][1].color = self.colors['outer_field']
                    else:
                        display_objects['agent-{}'.format(i_agent)][1].color = self.colors['agent-1']
            return drawing_util.render_visual_state({'camera': self.camera,
                                                     'display_objects': display_objects},
                                                    info_values,
                                                    pixels_per_worldunit=self.pixels_per_worldunit)


    @property
    def observation(self):
        """
        OpenAI Gym Observation
        :return:
            List of observations
        """
        observations = []
        for i_agent, agent in enumerate(self.agents):
            observation = self.render(mode='rgb_array', agent_id=i_agent)
            img = Image.fromarray(observation)
            img = img.resize(INPUT_SHAPE)  # .convert('L')  # resize and convert to grayscale
            processed_observation = np.array(img)
            observation = processed_observation.astype('uint8')
            observations.append(observation)

        return observations

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
    nb_agents = 2
    learning = decentral_learning
    nb_machine_types = 2
    nb_tasks = 3
    field_with = field_height = 6
    env = Smartfactory(nb_agents=nb_agents,
                                field_width=field_with,
                                field_height=field_height,
                                rewards=[1, 5],
                                learning=learning,
                                contracting=0,
                                nb_machine_types=nb_machine_types,
                                nb_tasks=nb_tasks)
    episodes = 1
    episode_steps = 100
    frames_a1 = []
    frames_a2 = []
    combined_frames = []

    for i_episode in range(episodes):

        observations = env.reset()
        info_values = [{'reward': 0.0,
                        'action': -1,
                        'signalling': -1} for _ in range(env.nb_agents)]
        frames_a1.append(env.render(mode='rgb_array', info_values=info_values[0], agent_id=0))
        frames_a2.append(env.render(mode='rgb_array', info_values=info_values[1], agent_id=1))
        combined_frames.append(np.append(frames_a1[-1], frames_a2[-1], axis=0))

        for i_ag in range(nb_agents):
            scipy.misc.toimage(observations[i_ag], cmin=0.0, cmax=...).save(
                'observations/new-outfile-{}-ag-{}.jpg'.format(0, i_ag))

        for i_step in range(1, episode_steps):
            actions = []
            if learning == decentral_learning:
                for i_agent in range(env.nb_agents):
                    actions.append(np.random.randint(0, env.nb_actions))
            elif learning == joint_learning:
                actions.append(np.random.randint(0, env.nb_actions))

            observations, rewards, done, _ = env.step(actions=actions)


            for i_ag in range(nb_agents):
                scipy.misc.toimage(observations[i_ag], cmin=0.0, cmax=...).save('observations/new-outfile-{}-ag-{}.jpg'.format(i_step, i_ag))


            if learning == joint_learning:
                rewards = [np.sum(rewards)]

            for i, agent in enumerate(env.agents):
                info_values[i]['reward'] = rewards[i]
                info_values[i]['action'] = actions[i]
                info_values[i]['signalling'] = agent.signalling

            frames_a1.append(env.render(mode='rgb_array', info_values=info_values[0], agent_id=0))
            frames_a2.append(env.render(mode='rgb_array', info_values=info_values[1], agent_id=1))
            combined_frames.append(np.append(frames_a1[-1], frames_a2[-1], axis=0))

            if done:
                break

    export_video('Smart-Factory.mp4', combined_frames, None)

if __name__ == '__main__':
    main()