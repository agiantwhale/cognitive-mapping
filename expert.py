import numpy as np
from scipy.misc import imresize
from scipy.ndimage.interpolation import shift, rotate
import networkx as nx
from top_view_renderer import EntityMap
from environment import get_entity_layer_path


class Expert(object):
    def _build_free_space_estimate(self, env_name):
        entity_map = EntityMap(get_entity_layer_path(env_name))
        wall_coordinates = frozenset((entity_map.height() - inv_row - 1, col)
                                     for col, inv_row in entity_map.wall_coordinates_from_string((1, 1)))
        self._walls = wall_coordinates

        self._env_name = env_name
        self._height = entity_map.height()
        self._width = entity_map.width()

        self._graph.clear()
        self._graph.add_nodes_from((row, col)
                                   for row in xrange(entity_map.height())
                                   for col in xrange(entity_map.width())
                                   if (row, col) not in wall_coordinates)

        for row in xrange(entity_map.height()):
            for col in xrange(entity_map.width()):
                if not self._graph.has_node((row, col)):
                    continue

                self._graph.add_edge((row, col), (row, col), weight=0)

                left = bottom = right = False

                # Left
                left_col = col - 1
                while self._graph.has_node((row, left_col)):
                    left = True
                    self._graph.add_edge((row, left_col), (row, col), weight=(col - left_col) * 100)
                    left_col -= 1

                # Bottom
                bottom_row = row + 1
                while self._graph.has_node((bottom_row, col)):
                    bottom = True
                    self._graph.add_edge((bottom_row, col), (row, col), weight=(bottom_row - row) * 100)
                    bottom_row += 1

                # Left
                right_col = col + 1
                while self._graph.has_node((row, right_col)):
                    right = True
                    self._graph.add_edge((row, right_col), (row, col), weight=(right_col - col) * 100)
                    right_col += 1

                # Bottom-Left
                bottom_row = row + 1
                left_col = col - 1
                if self._graph.has_node((bottom_row, left_col)) and bottom and left:
                    weight = int(np.sqrt(2) * (bottom_row - row) * 100)
                    self._graph.add_edge((bottom_row, left_col), (row, col), weight=weight)

                # Bottom-Right
                bottom_row = row + 1
                right_col = col + 1
                if self._graph.has_node((bottom_row, right_col)) and bottom and right:
                    weight = int(np.sqrt(2) * (bottom_row - row) * 100)
                    self._graph.add_edge((bottom_row, right_col), (row, col), weight=weight)

        self._weights = dict(nx.shortest_path_length(self._graph, weight='weight'))

    def _player_node(self, info):
        x, y = info.get('POSE')[:2]
        return int(self._height - y / 100), int(x / 100)

    def _goal_node(self, info):
        row, col = info.get('GOAL.LOC')
        return row - 1, col - 1

    def _node_to_game_coordinate(self, node):
        row, col = node
        return (col + 0.5) * 100, (self._height - row - 0.5) * 100

    def __init__(self):
        self._graph = nx.Graph()
        self._weights = {}
        self._env_name = None

    def get_goal_map(self, info, estimate_size=128):
        goal_map = np.zeros((estimate_size, estimate_size))
        game_scale = 1 / (960. / estimate_size)
        block_scale = int(100 * game_scale / 2)

        player_pos, player_rot = info.get('POSE')[:2], info.get('POSE')[4]
        goal_pos = np.array(self._node_to_game_coordinate(self._goal_node(info)))
        delta_pos = (goal_pos - player_pos) * game_scale
        # delta_angle = np.arctan2(delta_pos[1], delta_pos[0]) - player_rot

        c, s = np.cos(player_rot), np.sin(player_rot)
        rot_mat = np.array([[c, s], [-s, c]])
        x, y = np.dot(rot_mat, delta_pos).astype(np.int32)
        w = int(estimate_size / 2) + x
        h = int(estimate_size / 2) - y

        goal_map[h - block_scale:h + block_scale, w - block_scale:w + block_scale] = 1

        return np.expand_dims(goal_map, axis=2)

    def get_free_space_map(self, info, estimate_size=128):
        image = np.zeros((estimate_size, estimate_size), dtype=np.uint8) * 255
        game_scale = 1 / (960. / estimate_size)
        block_scale = 100 * game_scale

        for row, col in self._walls:
            w = int(col * block_scale)
            h = int((row - self._height) * block_scale)
            size = int(block_scale)
            w_end = w + size if (w + size) < estimate_size else estimate_size
            h_end = h + size if (h + size) != 0 else estimate_size
            image[h: h_end, w: w_end] = 255

        player_pos, player_rot = info['POSE'][:2], info['POSE'][4]
        w, h = player_pos * game_scale

        w -= estimate_size / 2
        h -= estimate_size / 2

        image = shift(image, [h, -w])
        print np.rad2deg(player_rot)
        image = rotate(image, -1 * np.rad2deg(player_rot))

        h, _ = image.shape
        crop_size = int((h - estimate_size) / 2)
        if crop_size > 0:
            image = image[crop_size:-crop_size, crop_size:-crop_size]

        image = imresize(image, size=(estimate_size, estimate_size))
        assert image.shape[0] == estimate_size

        return image

    def get_optimal_action(self, info):
        if self._env_name != info['env_name']:
            self._build_free_space_estimate(info['env_name'])

        player_pose = info.get('POSE')
        player_x, player_y, player_angle = player_pose[0], player_pose[1], player_pose[4]

        goal_node = self._goal_node(info)

        get_norm_angle = lambda angle: np.arctan2(np.sin(angle), np.cos(angle))
        get_game_angle = lambda x, y: np.arctan2(y - player_y, x - player_x)
        get_node_angle = lambda node: get_game_angle(*self._node_to_game_coordinate(node))
        node_criterion = lambda node: self._weights[node][goal_node] + \
                                      get_norm_angle(get_node_angle(node) - player_angle) / np.pi

        optimal_node = min(self._graph.neighbors(self._player_node(info)), key=node_criterion)

        action = np.zeros(4)

        if self._player_node(info) == goal_node:
            action[2] = 1
            return action

        angle_delta = get_norm_angle(get_node_angle(optimal_node) - player_angle)

        if abs(angle_delta) < np.deg2rad(7.5):
            action[2] = 1
        else:
            if angle_delta < 0:
                action[0] = 1
            else:
                action[1] = 1

        return action

    @property
    def entity_layer_name(self):
        return self._entity_layer_name
