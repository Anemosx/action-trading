from common_utils.utils import export_video
from rl.core import Processor
import scipy.misc

import gym
import gym.spaces
from collections import namedtuple
from copy import deepcopy
import common_utils.drawing_util as drawing_util
from common_utils.drawing_util import Camera, render_all_agents
import numpy as np
import Box2D
from Box2D import b2PolygonShape, b2FixtureDef, b2TestOverlap, b2Transform, b2Rot, b2ChainShape
from PIL import Image
import copy

INPUT_SHAPE = (84, 84)


class Agent:

    AgentState = namedtuple('AgentState', 'transform')

    def __init__(self, world, env, index, position, type):
        self.world = world
        self.index = index

        colors = [(0.6078431372549019, 0.34901960784313724, 0.7137254901960784),
                  (0.20392156862745098, 0.596078431372549, 0.8588235294117647),
                  (0.996078431372549, 0.7019607843137254, 0.03137254901960784),
                  (0.9058823529411765, 0.2980392156862745, 0.23529411764705882),
                  (0.1803921568627451, 0.8, 0.44313725490196076)]
        if type == 1:
            self.color = env.colors['agent_1']
        elif type == 2:
            self.color = env.colors['agent_2']
        else:
            self.color = env.colors['agent_3']

        self.type = type
        self.position = position
        self.agent_vertices = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        self.body = self.world.CreateDynamicBody(position=position,
                                                 angle=0,
                                                 angularDamping=0.6,
                                                 linearDamping=3.0,
                                                 shapes=[b2PolygonShape(vertices=self.agent_vertices)],
                                                 shapeFixture=b2FixtureDef(density=0.2))

        self.camera = Camera(pos=position, fov_dims=(3, 3))
        self.reset(position=self.position)

    def reset(self, position):
        self.body.position = position

    def get_state(self):
        ag_state = Agent.AgentState(transform=b2Transform(self.body.position, b2Rot(self.body.angle)))
        return ag_state

    def set_state(self, ag_state):
        self.body.transform = (ag_state.transform.position, ag_state.transform.angle)


class Checkpoint:

    State = namedtuple('CheckpointState', 'color visit_count')

    def __init__(self, world, env, index, position, nb_agents, food):
        self.world = world
        self.index = index
        self.vertices = [(-1, -1), (0, -1), (0, 0), (-1, 0)]
        self.shape = b2PolygonShape(vertices=self.vertices)
        colors = [(0.14248366013071898, 0.41730103806228375, 0.68335255670895811),
                  (0.32349096501345631, 0.61491733948481353, 0.78546712802768159),
                  (0.65490196078431362, 0.81437908496732025, 0.89411764705882346),
                  (0.88389081122645141, 0.92848904267589394, 0.95301806997308725),
                  (0.98200692041522486, 0.90618992695117262, 0.8615916955017302),
                  (0.96862745098039227, 0.71764705882352964, 0.60000000000000031),
                  (0.86228373702422145, 0.42952710495963098, 0.3427143406382161),
                  (0.71188004613610145, 0.12179930795847749, 0.18169934640522878)]
        if food:
            self.color = env.colors['food']
        else:
            self.color = env.colors['checkpoint']
        self.checkpoint_reward = 1.0
        self.position = position
        self.nb_agents = nb_agents
        self.visit_count = np.zeros(self.nb_agents)
        self.food = food

    def reset(self, env, index, display_objects, food):
        colors = [(0.14248366013071898, 0.41730103806228375, 0.68335255670895811),
                  (0.32349096501345631, 0.61491733948481353, 0.78546712802768159),
                  (0.65490196078431362, 0.81437908496732025, 0.89411764705882346),
                  (0.88389081122645141, 0.92848904267589394, 0.95301806997308725),
                  (0.98200692041522486, 0.90618992695117262, 0.8615916955017302),
                  (0.96862745098039227, 0.71764705882352964, 0.60000000000000031),
                  (0.86228373702422145, 0.42952710495963098, 0.3427143406382161),
                  (0.71188004613610145, 0.12179930795847749, 0.18169934640522878)]
        if food:
            self.color = env.colors['food']
        else:
            self.color = env.colors['checkpoint']
        display_objects['checkpoint{}'.format(index)][1].color = self.color
        self.visit_count = np.zeros(self.nb_agents)
        self.food = food

    def get_state(self):
        ch_state = Checkpoint.State(color=deepcopy(self.color),
                                    visit_count=deepcopy(self.visit_count))
        return ch_state

    def set_state(self, ch_state):
        self.color = deepcopy(ch_state.color)
        self.visit_count = deepcopy(ch_state.visit_count)


class MAWicksellianTriangle(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    State = namedtuple('MAWicksellianTriangle', 'agent_states checkpoint_states i_step')

    class MAWicksellianTriangleProcessor(Processor):
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
            return np.clip(reward, -1., 1.)

    def __init__(self, nb_agents, field_width, field_height):
        """

        :rtype: observation
        """
        self.world = Box2D.b2World(gravity=(0, 0))
        self.i_step = 0

        self.observation_space = gym.spaces.Box(0.0, 1.1, shape=(84, 84, 3))

        self.velocity_iterations = 6
        self.position_iterations = 2

        self.dt = 1.0 / 15
        self.agent_restitution = 0.5
        self.agent_density = 1.0

        self.colors = {
            'agent_1': (0.20392156862745098, 0.596078431372549, 0.8588235294117647),
            'agent_2': (0.9058823529411765, 0.2980392156862745, 0.23529411764705882),
            'agent_3': (0.96862745098039227, 0.71764705882352964, 0.60000000000000031),
            'outer_field':(0.5843137254901961, 0.6470588235294118, 0.6509803921568628),
            'field': (0.13, 0.15, 0.14, 1.0),
            'checkpoint': (0.13, 0.15, 0.14, 1.0),
            'food': (0.96862745098039227, 0.71764705882352964, 0.60000000000000031)
        }

        self.actions = {'up': [0.0, 1.0], 'down': [0.0, -1.0],
                        'left': [-1.0, 0.0], 'right': [1.0, 0.0]}
        self.action_space = gym.spaces.Discrete(n=len(self.actions))

        self.info = {'setting': None,
                     'episode': None,
                     'return_a0': None,
                     'return_a1': None
                     }

        self.pixels_per_worldunit = 25
        self.camera = Camera(pos=(0, 0), fov_dims=(25, 25))
        self.display_objects = dict()
        self.field_width = field_width
        self.field_height = field_height
        self.edge_margin = .2

        # Agents
        self.nb_agents = nb_agents
        self.agents = []
        self.agent_positions = [[(-(self.field_width-(self.field_width/2)-1))+column, (self.field_height/2)-row] for row in range(self.field_height) for column in range(self.field_width)]
        self.starting_positions = np.random.choice(self.field_width*self.field_height, self.nb_agents, replace=False)

        for i in range(self.nb_agents):
            agent = Agent(world=self.world,
                          env=self,
                          index=i,
                          position=self.agent_positions[self.starting_positions[i]],
                          type=np.random.randint(3))
            drawing_util.add_polygon(self.display_objects, agent.body, agent.agent_vertices,
                                     name='agent-{}'.format(i),
                                     drawing_layer=1,
                                     color=agent.color)
            self.agents.append(agent)

        self.field_vertices = []
        self.edge_vertices = []

        self._create_field()
        self.reset()

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

        self.i_step = 0

        self.starting_positions = np.random.choice(self.field_width * self.field_height, self.nb_agents, replace=False)

        for i_ag, agent in enumerate(self.agents):
            agent.reset(position=self.agent_positions[self.starting_positions[i_ag]])

        return self.observation

    def get_state(self):
        agent_states = []
        checkpoint_states = []
        for agent in self.agents:
            state = agent.get_state()
            agent_states.append(state)

        for chkp in self.checkpoints:
            state = chkp.get_state()
            checkpoint_states.append(state)

        state = MAWicksellianTriangle.State(agent_states=agent_states, checkpoint_states=checkpoint_states, i_step=self.i_step)

        return state

    def set_state(self, dk_state):
        for agent, ag_state in zip(self.agents, dk_state.agent_states):
            agent.set_state(ag_state=ag_state)
        for checkpoint, ch_state in zip(self.checkpoints, dk_state.checkpoint_states):
            checkpoint.set_state(ch_state=ch_state)
            self.display_objects['checkpoint{}'.format(checkpoint.index)][1].color = checkpoint.color
        self.i_step = dk_state.i_step
        # self.update_checkpoint_color()

    def step(self, actions):
        """
        :param actions: the list of agent actions
        :type actions: list
        """
        ma_actions = ['up', 'down', 'left', 'right']
        actions = [ma_actions[i] for i in actions]
        info = copy.deepcopy(self.info)
        rewards = np.zeros(self.nb_agents)
        done = False
        for i_agent in range(self.nb_agents):
            self.set_position(i_agent, self.agents[i_agent], actions[i_agent])

        if self.i_step >= 25 and not done:
            done = True
            rew = self.get_rewards()
            rewards += rew
            info['return_a0'] = rewards[0]
            info['return_a1'] = rewards[1]

        self.i_step += 1
        return self.observation, rewards, done, info

    def set_position(self, i_agent, agent, action):
        ac = self.actions[action]
        if action == 'up':
            if agent.body.transform.position.y + (1 * ac[1]) <= self.field_vertices[2][1]:
                agent.body.transform = ((agent.body.transform.position.x + (1.0 * ac[0]),
                                         agent.body.transform.position.y + (1.0 * ac[1])),
                                        0)
        if action == 'down':
            if agent.body.transform.position.y + (1 * ac[1]) > self.field_vertices[0][1]:
                agent.body.transform = ((agent.body.transform.position.x + (1.0 * ac[0]),
                                         agent.body.transform.position.y + (1.0 * ac[1])),
                                        0)
        if action == 'left':
            if agent.body.transform.position.x + (1 * ac[0]) > self.field_vertices[0][0]:
                agent.body.transform = ((agent.body.transform.position.x + (1.0 * ac[0]),
                                         agent.body.transform.position.y + (1.0 * ac[1])),
                                        0)
        if action == 'right':
            if agent.body.transform.position.x + (1 * ac[0]) <= self.field_vertices[1][0]:
                agent.body.transform = ((agent.body.transform.position.x + (1.0 * ac[0]),
                                         agent.body.transform.position.y + (1.0 * ac[1])),
                                        0)
        agent.camera.pos[0] = agent.body.transform.position.x - 0.5
        agent.camera.pos[1] = agent.body.transform.position.y - 0.5

    def get_rewards(self):

        rewards = np.zeros(self.nb_agents)

        return rewards

    def update_checkpoint_color(self):
        for checkpoint in self.checkpoints:
            if checkpoint.visit_count >= self.nb_agents:
                checkpoint.color = (0.6, 0.6, 0.6, 0.3)
                self.set_checkpoint_color(checkpoint.index, checkpoint.color)

    def set_checkpoint_color(self, i, color):
        self.display_objects['checkpoint{}'.format(i)][1].color = color

    def render(self, mode='human', close=False, info_values=None, camera=None):
        if mode == 'rgb_array':
            if info_values is not None:
                return drawing_util.render_visual_state_with_information({'camera': self.camera,
                                                         'display_objects': self.display_objects}, info_values,
                                                        pixels_per_worldunit=self.pixels_per_worldunit)
            elif camera is not None:
                return drawing_util.render_visual_state({'camera': camera,
                                                         'display_objects': self.display_objects},
                                                        pixels_per_worldunit=self.pixels_per_worldunit)
            else:
                return drawing_util.render_visual_state({'camera': self.camera,
                                                         'display_objects': self.display_objects},
                                                        pixels_per_worldunit=self.pixels_per_worldunit)

    @property
    def observation(self):
        observations = []
        for agent in self.agents:
            observation = self.render(mode='rgb_array', camera=agent.camera)
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


def main():
    nb_agents = 36
    log_dir = '../experiments/experiments/20181215-18-18-52'

    env = MAWicksellianTriangle(nb_agents=nb_agents, field_width=30, field_height=30)
    episodes = 1
    episode_steps = 100
    frames = []

    for i_episode in range(episodes):

        _ = env.reset()

        observations = []
        for agent in env.agents:
            obs = env.render(mode='rgb_array', camera=agent.camera)
            observations.append(obs)

        rewards = np.zeros(2)
        done = False
        info = None
        info_values = [{'reward': 0.0} for _ in range(env.nb_agents)]

        for i in range(episode_steps):

            actions = []
            for i_agent in range(env.nb_agents):
                actions.append(np.random.randint(0, 4))

            observations, rewards, done, _ = env.step(actions=actions)

            scipy.misc.toimage(observations[0], cmin=0.0, cmax=...).save('observations/new-outfile-{}.jpg'.format(i))

            for i_r, reward in enumerate(rewards):
                info_values[i_r]['reward'] = reward

            observations = []
            for agent in env.agents:
                obs = env.render(mode='rgb_array', camera=agent.camera)
                observations.append(obs)

            frames.append(render_all_agents(env, info_values=info_values, observations=observations))

            if done:
                break

    export_video('MAWicksellianTriangle-test.mp4', frames, None)

if __name__ == '__main__':
    main()
