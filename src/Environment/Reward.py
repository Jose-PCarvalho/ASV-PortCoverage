from src.Environment.Actions import Events
from src.Environment.State import State


class RewardParams:
    def __init__(self):
        self.blocked_reward = -1
        self.repeated_field_reward = -1
        self.new_tile_reward = 1
        self.map_complete = 0  # max_size ** 2 - scaling ** 2
        self.timeout = 0  # scaling ** 2
        self.close_to_wall_reward = 1.0
        self.repeated_action_reward = 1.0
        self.finished_row_col = 1.0
        self.repeating_two_moves = -1.0


class GridRewards:
    def __init__(self, params):
        self.number_agents = params.state_params.number_agents
        self.last_remaining_potential = None
        self.last_closest = None
        self.params = params.reward_params
        self.cumulative_reward = [0 for _ in range(self.number_agents)]
        self.overlap = [0 for _ in range(self.number_agents)]
        self.steps = 0
        self.total_steps = None
        self.remaining = None
        self.closest_dist = None
        self.closest_cell = None
        self.stuck = [0 for _ in range(self.number_agents)]
        self.optimal_steps = None

    def get_cumulative_reward(self):
        return sum(self.cumulative_reward) / len(self.cumulative_reward)

    def get_overlap(self):
        overlap = [self.overlap[i] / (self.steps -self.overlap[i]) for i in range(self.number_agents)]
        return sum(overlap) / len(overlap)  # self.overlap / (self.steps - self.overlap)
    
    def get_time_save(self):
        return self.steps/self.optimal_steps

    def reset(self, state: State):
        self.cumulative_reward = [0 for _ in range(self.number_agents)]
        self.overlap = [0 for _ in range(self.number_agents)]
        self.steps = 0
        self.params.scaling_factor = 1  # scaling ** 2
        self.total_steps = state.remaining
        self.remaining = state.remaining
        self.last_remaining_potential = -self.remaining  # / self.total_steps
        self.closest_dist = [-state.local_map.min_manhattan_distance(state.position[i].get_position())[0] for i in
                        range(self.number_agents)]               
        self.closest_cell = [state.local_map.min_manhattan_distance(state.position[i].get_position())[1] for i in
                        range(self.number_agents)]                
        self.stuck = [0 for _ in range(self.number_agents)]
        self.optimal_steps = state.optimal_steps

    def compute_reward(self, events, state: State):
        r = [0 for _ in range(self.number_agents)]
        self.steps += 1
        self.remaining = state.remaining
        for i, event in enumerate(events):
            new_remaining_potential = - self.remaining  # / self.total_steps
            new_closest_dist , new_closest_cell = state.local_map.min_manhattan_distance(state.position[i].get_position())
            new_closest_dist *=-1
            diff = (new_remaining_potential - self.last_remaining_potential)
            r[i] += (new_remaining_potential - self.last_remaining_potential) * 1
            if diff != 0:
                self.stuck[i] = 0
            else:
                # r += self.params.repeated_field_reward
                if all(new_closest_cell == self.closest_cell[i]):
                    new_dist = -len(state.local_map.dijkstra_search(state.position[i].get_position(),(new_closest_cell[0],new_closest_cell[1])))
                    old_dist = -len(state.local_map.dijkstra_search(state.position[i].get_position(),(self.closest_cell[i][0],self.closest_cell[i][1])))
                    dist = min(0.5*(new_dist - old_dist),0.5)
                else:
                    dist = 0
                r[i] += max(0,0.5 * (new_closest_dist - self.closest_dist[i]),dist)
                self.overlap[i] += 1
                self.stuck[i] += 1
            if Events.BLOCKED in event:
                r[i] += self.params.blocked_reward
            if Events.FINISHED in event:
                r[i] += self.params.map_complete
            if Events.TIMEOUT in event:
                r[i] += self.params.timeout
            if Events.WAITED in event:
                if self.number_agents==1:
                    r[i]+=-1
                else:
                    r[i] += 0.25
            r[i] += self.params.repeated_field_reward
            if self.number_agents > 1:
                r[i] += (0)/(self.number_agents-1)*(events.count(Events.NEW) - 1*(Events.NEW in event))
            self.last_remaining_potential = new_remaining_potential
            self.closest_dist[i] , self.closest_cell[i] = new_closest_dist, new_closest_cell
            self.cumulative_reward[i] += r[i]

        return r
