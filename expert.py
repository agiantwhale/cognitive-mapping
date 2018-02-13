import numpy as np
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

                left = bottom = right = False

                # Left
                if self._graph.has_node((row, col - 1)):
                    left = True
                    self._graph.add_edge((row, col - 1), (row, col), weight=1)

                # Bottom
                if self._graph.has_node((row + 1, col)):
                    bottom = True
                    self._graph.add_edge((row + 1, col), (row, col), weight=1)

                # Left
                if self._graph.has_node((row, col + 1)):
                    right = True
                    self._graph.add_edge((row, col + 1), (row, col), weight=1)

                # Bottom-Left
                if self._graph.has_node((row + 1, col - 1)) and bottom and left:
                    self._graph.add_edge((row + 1, col - 1), (row, col), weight=np.sqrt(2))

                # Bottom-Right
                if self._graph.has_node((row + 1, col + 1)) and bottom and right:
                    self._graph.add_edge((row + 1, col + 1), (row, col), weight=np.sqrt(2))

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

    def get_goal_map(self, info, estimate_size=64):
        goal_map = np.zeros((estimate_size, estimate_size))

        player_pos, player_rot = info.get('POSE')[:2], info.get('POSE')[4]
        goal_pos = np.array(self._node_to_game_coordinate(self._goal_node(info)))
        delta_pos = (goal_pos - player_pos) * estimate_size / 1400.
        # delta_angle = np.arctan2(delta_pos[1], delta_pos[0]) - player_rot

        c, s = np.cos(player_rot), np.sin(player_rot)
        rot_mat = np.array([[c, s], [-s, c]])
        x, y = np.dot(rot_mat, delta_pos).astype(np.int32)
        w = int(estimate_size / 2) + x
        h = int(estimate_size / 2) - y

        goal_map[h - 1:h + 1, w-1:w+1] = 10

        return goal_map

    def get_optimal_action(self, info):
        if self._env_name != info['env_name']:
            self._build_free_space_estimate(info['env_name'])

        action = np.zeros(4)

        optimal_node = min(self._graph.neighbors(self._player_node(info)),
                           key=lambda neighbor: self._weights[neighbor][self._goal_node(info)])

        if self._player_node(info) == self._goal_node(info):
            optimal_node = self._goal_node(info)

        player_pose = info.get('POSE')
        player_x, player_y, player_angle = player_pose[0], player_pose[1], player_pose[4]
        optimal_x, optimal_y = self._node_to_game_coordinate(optimal_node)

        optimal_angle = np.arctan2(optimal_y - player_y, optimal_x - player_x)

        angle_delta = optimal_angle - player_angle

        if abs(angle_delta) < 0.15 or abs(abs(angle_delta) - 2 * np.pi) < 0.15:
            action[2] = 1
        else:
            if abs(angle_delta) >= np.pi:
                if angle_delta > 0:
                    action[0] = 1
                else:
                    action[1] = 1
            else:
                if angle_delta < 0:
                    action[0] = 1
                else:
                    action[1] = 1

        return action

    @property
    def entity_layer_name(self):
        return self._entity_layer_name
