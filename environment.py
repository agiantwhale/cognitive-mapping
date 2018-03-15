import os
import inspect
import deepmind_lab as dl
import deepmind_lab_gym as dlg
import multiprocdmlab as mpdmlab
import numpy as np
from random import shuffle

DEEPMIND_RUNFILES_PATH = os.path.dirname(inspect.getfile(dl))
DEEPMIND_SOURCE_PATH = os.path.abspath(DEEPMIND_RUNFILES_PATH + '/..' * 5)
dl.set_runfiles_path(DEEPMIND_RUNFILES_PATH)


def get_entity_layer_path(entity_layer_name):
    global DEEPMIND_RUNFILES_PATH, DEEPMIND_SOURCE_PATH
    mode, size, num = entity_layer_name.split('-')
    path_format = '{}/assets/entityLayers/{}/{}/entityLayers/{}.entityLayer'
    path = path_format.format(DEEPMIND_SOURCE_PATH, size, mode, num)

    return path


def get_game_environment(mapname='training-09x09-0127', mode='training', multiproc=False,
                         random_spawn=True, random_goal=True, apple_prob=0.9):
    maplist = mapname.split(',')
    shuffle(maplist)
    mapstrings = ','.join(open(get_entity_layer_path(m)).read() for m in maplist)

    params = {
        'level_script': 'random_mazes',
        'config': dict(width=84, height=84, fps=30
                       , rows=9
                       , cols=9
                       , mode=mode
                       , num_maps=1
                       , withvariations=True
                       , random_spawn_random_goal='True'
                       , goal_characters='G' if not random_goal else 'GAP'
                       , spawn_characters='P' if not random_spawn else 'GAP'
                       , chosen_map=mapname
                       , mapnames=mapname
                       , mapstrings=mapstrings
                       , apple_prob=apple_prob
                       , episode_length_seconds=5),
        'action_mapper': dlg.ActionMapperDiscrete,
        'enable_depth': True,
        'additional_observation_types': ['GOAL.LOC', 'SPAWN.LOC', 'POSE', 'GOAL.FOUND']
    }

    if multiproc:
        params['deepmind_lab_class'] = dlg.DeepmindLab
        params['mpdmlab_workers'] = 1
        env = mpdmlab.MultiProcDeepmindLab(**params)
    else:
        env = dlg.DeepmindLab(**params)

    return env


def calculate_egomotion(previous_pose, current_pose):
    previous_pos, previous_angle = previous_pose[:2], previous_pose[4]
    current_pos, current_angle = current_pose[:2], current_pose[4]

    rotation = current_angle - previous_angle
    abs_translation = current_pos - previous_pos
    abs_angle = np.arctan2(abs_translation[1], abs_translation[0])
    delta_angle = abs_angle - current_angle
    translation = np.array([np.cos(delta_angle), np.sin(delta_angle)]) * np.linalg.norm(abs_translation)

    return translation.tolist() + [rotation]
