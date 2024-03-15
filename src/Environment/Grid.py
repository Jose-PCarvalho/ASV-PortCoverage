import numpy as np
from math import ceil
import heapq


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return not self.elements

    def put(self, item, priority: float):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class GridMap:
    def __init__(self, a=None, start=None):

        if start is not None:
            self.map = {start[0]: []}
            self.visited_list = []
            for tile in start:
                self.new_tile(tile)
                self.visited_list.append(tile)
        else:
            self.map = {}
            self.visited_list = []
        self.obstacle_list = []

        if a is not None:
            self.height = a.shape[0]
            self.width = a.shape[1]
            # self.map_array = np.zeros((self.height, self.width), dtype=int)
            for i in range(self.width):
                for j in range(self.height):
                    self.new_tile((i, j), a[i, j] == -1)
            for t in self.getTiles():
                if not self.map[t] and t not in self.obstacle_list:
                    self.obstacle_list.append(t)

        else:
            x = max([start[i][0] for i in range(len(start))])
            y = max([start[i][1] for i in range(len(start))])
            self.height = max(x + 1, y + 1, 2)
            self.width = self.height
        self.map_array = self.graph_to_array()

    @staticmethod
    def adjacentTiles(tile):
        return [(tile[0] + 1, tile[1]), (tile[0] - 1, tile[1]), (tile[0], tile[1] + 1), (tile[0], tile[1] - 1)]

    def getTiles(self):
        return list(self.map.keys())

    def new_tile(self, tile, obstacle=False):
        if tile not in self.getTiles():
            if tile[0] >= self.height:
                self.height = tile[0] + 1
                self.width = self.height

            elif tile[1] >= self.width:
                self.width = tile[1] + 1
                self.height = self.width

            self.map[tile] = []
            if not obstacle:
                adjacent = self.adjacentTiles(tile)
                for adj in adjacent:
                    if adj in set(self.getTiles()).difference(set(self.obstacle_list)):
                        self.map[tile].append(adj)
                        self.map[adj].append(tile)
            else:
                self.obstacle_list.append(tile)
            self.map_array = self.graph_to_array()

    def print_graph(self):
        tiles = self.getTiles()
        for t in tiles:
            print(t, ": ", self.map[t])
        print("end\n")

    def visit_tile(self, tile):
        if tile not in self.visited_list:
            self.visited_list.append(tile)
            self.map_array[:, tile[0], tile[1]] = [255, 0, 0, 0]

    def graph_to_array(self):
        a = np.zeros((4, self.height, self.width),
                     dtype=np.uint8)  # 0 -> visited, #2 -> obstacles 3->not-seen
        for i in range(self.height):
            for j in range(self.width):
                tile = (i, j)
                if tile in self.visited_list:
                    a[:, i, j] = [255, 0, 0, 0]
                elif tile in self.obstacle_list:
                    a[:, i, j] = [0, 255, 0, 0]
                elif tile in (self.getTiles()) and tile not in (set(self.visited_list).union(set(self.obstacle_list))):
                    a[:, i, j] = [0, 0, 255, 0]
        return a

    def bresenham(self, x0, y0, x1, y1):
        is_steep = abs(y1 - y0) > abs(x1 - x0)
        # Swap coordinates if the line is steep
        if is_steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        # Swap endpoints if necessary to make sure we are always iterating left to right
        swap = False
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            swap = True
        # Recalculate differentials
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx // 2
        ystep = 1 if y0 < y1 else -1
        # Initial y coordinate
        y = y0
        to_visit = []
        for x in range(x0, x1 + 1):
            if is_steep:
                to_visit.append((y, x))
            else:
                to_visit.append((x, y))

            error -= dy
            if error < 0:
                y += ystep
                error += dx
        if swap:
            to_visit.reverse()
        for t in to_visit:
            if (t[0],t[1]) in self.obstacle_list or (t[0], t[1]) not in self.getTiles():
                break
            self.visit_tile(t)

    def laser_scanner(self, tile, full_map, r):
        limit = []
        for i in range(-r, r + 1):
            limit.append((i, -r))
            limit.append((i, r))
            limit.append((r, i))
            limit.append((-r, i))
        for l in limit:
            self.bresenham(tile[0], tile[1], tile[0] + l[0], tile[1] + l[1])


    def camera(self, tile, full_map, r):
        tiles = []
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                tiles.append((tile[0] + i, tile[1] + j))

        full_map_tiles = full_map.getTiles()
        local_map_tiles = self.getTiles()
        for t in tiles:
            if t in full_map_tiles and t not in local_map_tiles and all(cord >= 0 for cord in t):
                self.new_tile(t, obstacle=t in full_map.obstacle_list)

    def update_agent_position(self, old_position, new_position):
        self.map_array[3, old_position[0], old_position[1]] = 0
        self.map_array[3, new_position[0], new_position[1]] = 255

    def center_map(self, position):
        new_size = max(37, max(self.height, self.width) * 2 - 1)
        # calculate the center index of the new array
        center_index = new_size // 2

        # create a new array of zeros with the desired size
        new_arr = np.zeros((4, new_size , new_size ), dtype=np.uint8)
        new_arr[1, :, :] = 255
        # calculate the indices of the original array that should be copied to the new array
        start_i = center_index - position[0]
        end_i = start_i + self.map_array.shape[1]
        start_j = center_index - position[1]
        end_j = start_j + self.map_array.shape[2]
        # copy the original array to the center of the new array
        new_arr[:, start_i:end_i, start_j:end_j] = self.map_array
        tiles_to_go = np.zeros((4, 4), dtype=np.int32)
        # if new_size > 37:
        #     start_index = center_index - 18 + 2  # (37 - 1) // 2
        #     end_index = center_index + 19 + 2  # (37 + 1) // 2
        #     missing_up = start_i - start_index
        #     missing_down = end_index - end_i
        #     missing_left = start_j - start_index
        #     missing_right = end_index - end_j
        #
        #     up_edge = np.zeros((4, 41))
        #     up_edge[1, :] = 255
        #     down_edge = np.zeros((4, 41))
        #     down_edge[1, :] = 255
        #     left_edge = np.zeros((4, 41))
        #     left_edge[1, :] = 255
        #     right_edge = np.zeros((4, 41))
        #     right_edge[1, :] = 255
        #
        #     if missing_up < 0:
        #         tiles_to_go[0][0] = np.count_nonzero(new_arr[2, start_i:start_index, :])
        #         tiles_to_go[0][1] = -1 * missing_up
        #         up_edge = compress_edge(np.mean(new_arr[:, start_i:start_index, :], axis=1), start_j, end_j,
        #                                 missing_left, missing_right)
        #     if missing_down < 0:
        #         tiles_to_go[1][0] = np.count_nonzero(new_arr[2, end_index:end_i, :])
        #         tiles_to_go[1][1] = -1 * missing_down
        #         down_edge = compress_edge(np.mean(new_arr[:, end_index:end_i, :], axis=1), start_j, end_j,
        #                                   missing_left, missing_right)
        #     if missing_left < 0:
        #         tiles_to_go[2][0] = np.count_nonzero(new_arr[2, :, start_j:start_index])
        #         tiles_to_go[2][1] = -1 * missing_left
        #         left_edge = compress_edge(np.mean(new_arr[:, :, start_j:start_index], axis=2), start_i, end_i,
        #                                   missing_up, missing_down)
        #     if missing_right < 0:
        #         tiles_to_go[3][0] = np.count_nonzero(new_arr[2, :, end_index:end_j])
        #         tiles_to_go[3][1] = -1 * missing_right
        #         right_edge = compress_edge(np.mean(new_arr[:, :, end_index:end_j], axis=2), start_i, end_i,
        #                                    missing_up, missing_down)
        #     # Extract the 37x37 array
        #     inner_array = new_arr[:, start_index:end_index, start_index:end_index]
        #     new_arr = np.zeros((4, 41, 41), dtype=np.uint8)
        #     new_arr[1, :, :] = 255
        #     new_arr[:, 1, :], new_arr[:, -2, :], new_arr[:, :, 1], new_arr[:, :, -2] = up_edge, down_edge, left_edge \
        #         , right_edge
        #
        #     new_arr[:, 1, 1] = (up_edge[:, 1] + left_edge[:, 1]) // 2
        #
        #     new_arr[:, 1, -2] = (up_edge[:, -2] + right_edge[:, 1]) // 2
        #     new_arr[:, -2, 1] = (down_edge[:, 1] + left_edge[:, -2]) // 2
        #     new_arr[:, -2, -2] = (down_edge[:, -2] + right_edge[:, -2]) // 2
        #     new_arr[:, 2:-2, 2:-2] = inner_array

        return new_arr, tiles_to_go

    def fix_map(self, start):
        visited = self.dfs(start)
        non_obs = set(self.getTiles()).difference(self.obstacle_list)
        should_be_obs = non_obs.difference(visited)
        for t in should_be_obs:
            for neighbor in self.map[t]:
                self.map[neighbor].remove(t)
            self.map[t] = []
            self.obstacle_list.append(t)
        self.map_array = self.graph_to_array()

    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        visited.add(start)
        non_obs = set(self.getTiles()).difference(self.obstacle_list)
        l_non_obs = len(non_obs)
        if len(visited) == l_non_obs:
            return visited

        for neighbor in self.map[start]:
            if neighbor not in visited:
                self.dfs(neighbor, visited)

        return visited

    def min_manhattan_distance(self, agent_pos):
        not_seen_list = set(self.getTiles()).difference(set(self.visited_list).union(set(self.obstacle_list)))
        if len(not_seen_list) == 0:
            return 0, [0, 0]
        positions = np.array(list(not_seen_list))
        distances = positions - agent_pos
        distances = np.linalg.norm(distances, ord=1, axis=1)
        dist = min(distances) - 1
        dist_idx = np.argmin(distances)
        closest = positions[dist_idx]
        return dist, closest

    def path_min_manhattan(self, agent_pos):
        not_seen_list = set(self.getTiles()).difference(set(self.visited_list).union(set(self.obstacle_list)))
        if len(not_seen_list) == 0:
            return 0, [0, 0]
        positions = np.array(list(not_seen_list))
        distances = positions - agent_pos
        distances = np.linalg.norm(distances, ord=1, axis=1)
        indices = sorted(range(len(distances)), key=lambda x: distances[x])
        return positions, indices

    def dijkstra_search(self, start, finish):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not frontier.empty():
            current = frontier.get()

            if current == finish:
                break

            for next in self.map[current]:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost
                    frontier.put(next, priority)
                    came_from[next] = current

        return self.reconstruct_path(came_from, start, finish)

    @staticmethod
    def reconstruct_path(came_from, start, finish):
        current = finish
        path = []
        if finish not in came_from.keys():  # no path was found
            return []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()  # optional
        return path


def compress_edge(edge, start_i, end_i, begin_offset, end_offset):
    start_offset = min(0, begin_offset)
    finish_offset = min(0, end_offset)
    b_offset = max(begin_offset, 0)
    use_begin_edge = 0
    if begin_offset < 0:
        use_begin_edge = 1
        begin_edge = np.mean(edge[:, start_i:start_i - start_offset], axis=1)
    use_end_edge = 0
    if end_offset < 0:
        use_end_edge = 1
        end_edge = np.mean(edge[:, end_i + finish_offset:end_i], axis=1)
    new_edge = np.zeros((4, 41), dtype=np.uint16)
    new_edge[1, :] = 255
    edge = edge[:, (start_i - start_offset - use_begin_edge):(end_i + finish_offset + use_end_edge)]
    edge_size = edge.shape[1]
    new_edge[:, (b_offset + 2 - use_begin_edge):(b_offset + 2 + edge_size - use_begin_edge)] = edge
    if use_end_edge == 1:
        new_edge[:, -2] = end_edge
    if use_begin_edge == 1:
        new_edge[:, 1] = begin_edge
    new_edge[0, :][new_edge[0, :] > 0] = 255
    return new_edge
