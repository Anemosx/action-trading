import os
import json
import numpy as np
import common_utils.drawing_util
import moviepy.editor as mpy
import gizeh
from collections import OrderedDict
import argparse

def save_params(params, output_path):
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        json.dump(params, f)


def render_observation(observation, info_values, width, height, env, agent):

    surface = gizeh.Surface(width, height, (0, 0, 0, 1))

    frame = gizeh.polyline([(0, 0), (0, height), (0, height), (width, height), (width, height), (width, 0), (0, 0)],
                           stroke_width=5, stroke=(1, 1, 1, 0.8))
    frame.draw(surface)

    if info_values is not None:
        offset_h = (height - height / 7)
        offset_w = width - width / 2.5
        for i, (key, value) in enumerate(info_values.items()):
            if key == 'done':
                text = gizeh.text('{0}: {1}'.format(key, value), fontfamily="Helvetica", fontsize=12,
                                  fill=(1, 1, 1), xy=(offset_w, offset_h), angle=0, h_align='left')
            else:
                text = gizeh.text('{0}: {1:.2}'.format(key, value), fontfamily="Helvetica", fontsize=12,
                                  fill=(1, 1, 1), xy=(offset_w, offset_h), angle=0, h_align='left')
            text.draw(surface)
            offset_h += 12

    return surface.get_npimage()


def render(env, observations, rewards, done):
    frame = env.render(mode='rgb_array')
    width = frame.shape[1]
    height = frame.shape[0]

    infos = [OrderedDict([('reward', rewards[i_obs]),
                          ('done', done),
                          ]) for i_obs, observation in enumerate(observations)]
    rendered_observations = [frame]
    rendered_observations += [render_observation(observations[i_agent], infos[i_agent], width, height, env=env,
                                                 agent=agent) for i_agent, agent in enumerate(env.agents)]

    vert_frames = [np.append(rendered_observations[i], rendered_observations[i + 1], axis=0) for i in
                   range(0, env.nb_agents, 2)]
    frame = vert_frames[0]
    for i_vert_frame in range(1, len(vert_frames)):
        frame = np.append(frame, vert_frames[i_vert_frame], axis=1)

    return frame


def render_for_agent(env, observations, rewards, done, agent):
    frame = env.render(mode='rgb_array', agent=agent)
    width = frame.shape[1]
    height = frame.shape[0]

    infos = [OrderedDict([('reward', rewards[i_obs]),
                          ('done', done),
                          ]) for i_obs, observation in enumerate(observations)]
    rendered_observations = [frame]
    rendered_observations += [render_observation(observations[i_agent], infos[i_agent], width, height, env=env,
                                                 agent=agent) for i_agent, agent in enumerate(env.agents)]

    vert_frames = [np.append(rendered_observations[i], rendered_observations[i + 1], axis=0) for i in
                   range(0, env.nb_agents, 2)]
    frame = vert_frames[0]
    for i_vert_frame in range(1, len(vert_frames)):
        frame = np.append(frame, vert_frames[i_vert_frame], axis=1)

    return frame



def export_video(filename, frames, info_values=None):
    assert filename.endswith('.mp4')

    if info_values is None:
        combined_frames = frames
    else:
        info_screen_height = 30 * max(len(info) for info in info_values)
        combined_frames = [np.append(frame,
                                     drawing_util.render_info(info_value, (info_screen_height, frame.shape[1])),
                                     axis=0)
                           for frame, info_value in zip(frames, info_values)]

    clip = mpy.ImageSequenceClip(combined_frames, fps=30)
    clip.write_videofile(filename)


class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None for _ in range(maxlen)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class PathType(object):
    def __init__(self, exists=True, type='file', dash_ok=True):
        '''exists:
                True: a path that does exist
                False: a path that does not exist, in a valid parent directory
                None: don't care
           type: file, dir, symlink, None, or a function returning True for valid paths
                None: don't care
           dash_ok: whether to allow "-" as stdin/stdout'''

        assert exists in (True, False, None)
        assert type in ('file','dir','symlink',None) or hasattr(type,'__call__')

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string):
        error = argparse.ArgumentTypeError
        if string=='-':
            # the special argument "-" means sys.std{in,out}
            if self._type == 'dir':
                raise error('standard input/output (-) not allowed as directory path')
            elif self._type == 'symlink':
                raise error('standard input/output (-) not allowed as symlink path')
            elif not self._dash_ok:
                raise error('standard input/output (-) not allowed')
        else:
            e = os.path.exists(string)
            if self._exists==True:
                if not e:
                    raise error("path does not exist: '%s'" % string)

                if self._type is None:
                    pass
                elif self._type=='file':
                    if not os.path.isfile(string):
                        raise error("path is not a file: '%s'" % string)
                elif self._type=='symlink':
                    if not os.path.symlink(string):
                        raise error("path is not a symlink: '%s'" % string)
                elif self._type=='dir':
                    if not os.path.isdir(string):
                        raise error("path is not a directory: '%s'" % string)
                elif not self._type(string):
                    raise error("path not valid: '%s'" % string)
            else:
                if self._exists==False and e:
                    raise error("path exists: '%s'" % string)

                p = os.path.dirname(os.path.normpath(string)) or '.'
                if not os.path.isdir(p):
                    raise error("parent path is not a directory: '%s'" % p)
                elif not os.path.exists(p):
                    raise error("parent directory does not exist: '%s'" % p)

        return string