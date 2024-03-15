from __future__ import division
import numpy as np
import torch

# State_dtype = np.dtype([('frames', np.uint8, (4, 9, 9)), ('battery', np.int64)])
# blank_state = (np.zeros((4, 9, 9), dtype=np.uint8), 0)

Transition_dtype = np.dtype(
    [('timestep', np.int32), ('state', np.uint8, (4, 59, 59)), ('battery', np.int32), ('last_action', np.int64),
     ('orientation', np.int32), ('action', np.int32), ('reward', np.float32), ('nonterminal', np.bool_)])
blank_trans = (0, np.zeros((4, 59, 59), dtype=np.uint8), 0, 0, 0, 0, 0.0, False)


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2 ** (size - 1).bit_length() - 1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)  # Build structured array
        self.max = 1  # Initial max value to return (1 = 1^ω)

    # Updates nodes values from current tree
    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)

    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1))  # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(
            np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)]  # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]


class ReplayMemory():
    def __init__(self, args, capacity):
        self.device = args.device
        self.capacity = capacity
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_weight = args.priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = args.priority_exponent
        self.t = 0  # Internal episode timestep counter
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32,
                                           device=self.device)  # Discount-scaling vector for n-step returns
        self.transitions = SegmentTree(
            capacity)  # Store transitions in a wrap-around cyclic buffer within a sum tree for querying priorities

    # Adds state and action at time t, reward and terminal at time t + 1
    def append(self, state, battery, last_action, orientation, action, reward, terminal, truncated):
        for s , b, l_a, o , r , a in zip(state,battery,last_action,orientation,reward,action):
            s_ = torch.tensor(s[-1], dtype=torch.uint8, device=torch.device('cpu'))
            b_ = torch.tensor(b, dtype=torch.int32, device=torch.device('cpu'))
            l_a_ = torch.tensor(l_a[-1], dtype=torch.int32, device=torch.device('cpu'))
            o_ = torch.tensor(o[-1], dtype=torch.int32, device=torch.device('cpu'))
            self.transitions.append((self.t, s_, b_, l_a_, o_, a, r, not terminal),
                                    self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if terminal or truncated else self.t + 1  # Start new episodes with t = 0
    def _get_transitions(self, idxs):
        transition_idxs = np.arange(-self.history + 1, self.n + 1) + np.expand_dims(idxs, axis=1)
        transitions = self.transitions.get(transition_idxs)
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in range(self.history - 2, -1, -1):  # e.g. 2 1 0
            blank_mask[:, t] = np.logical_or(blank_mask[:, t + 1],
                                             transitions_firsts[:, t + 1])  # True if future frame has timestep 0
        for t in range(self.history, self.history + self.n):  # e.g. 4 5 6
            blank_mask[:, t] = np.logical_or(blank_mask[:, t - 1],
                                             transitions_firsts[:, t])  # True if current or past frame has timestep 0
        transitions[blank_mask] = blank_trans
        return transitions

    # Returns a valid sample from each segment
    def _get_samples_from_segments(self, batch_size, p_total):
        segment_length = p_total / batch_size  # Batch size number of segments, based on sum over all probabilities
        segment_starts = np.arange(batch_size) * segment_length
        valid = False
        while not valid:
            samples = np.random.uniform(0.0, segment_length,
                                        [batch_size]) + segment_starts  # Uniformly sample from within all segments
            probs, idxs, tree_idxs = self.transitions.find(
                samples)  # Retrieve samples from tree with un-normalised probability
            if np.all((self.transitions.index - idxs) % self.capacity > self.n) and np.all(
                    (idxs - self.transitions.index) % self.capacity >= self.history) and np.all(probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0
        # Retrieve all required transition data (from t - h to t + n)
        transitions = self._get_transitions(idxs)
        # Create un-discretised states and nth next states

        all_states = transitions['state']
        all_batteries = transitions['battery'].copy()
        all_last_actions = transitions['last_action'].copy()
        all_orientation = transitions['orientation'].copy()
        states = torch.tensor(all_states[:, :self.history], device=self.device, dtype=torch.float32) / 255
        next_states = torch.tensor(all_states[:, self.n:self.n + self.history], device=self.device,
                                   dtype=torch.float32) / 255  # Discrete actions to be used as index

        battery = torch.tensor(all_batteries[:, self.history - 1], device=self.device, dtype=torch.int32)
        next_battery = torch.tensor(all_batteries[:, -1], device=self.device, dtype=torch.int32)

        last_action = torch.tensor(all_last_actions[:, :self.history], device=self.device, dtype=torch.int64)
        next_last_action = torch.tensor(all_last_actions[:, self.n:self.n + self.history], device=self.device,
                                        dtype=torch.int64)

        orientation = torch.tensor(all_orientation[:, :self.history], device=self.device, dtype=torch.int64)
        next_orientation = torch.tensor(all_orientation[:, self.n:self.n + self.history], device=self.device,
                                       dtype=torch.int64)

        actions = torch.tensor(np.copy(transitions['action'][:, self.history - 1]), dtype=torch.int64,
                               device=self.device)
        # Calculate truncated n-step discounted returns R^n = Σ_k=0->n-1 (γ^k)R_t+k+1 (note that invalid nth next states have reward 0)
        rewards = torch.tensor(np.copy(transitions['reward'][:, self.history - 1:-1]), dtype=torch.float32,
                               device=self.device)
        R = torch.matmul(rewards, self.n_step_scaling)
        # Mask for non-terminal nth next states
        nonterminals = torch.tensor(transitions['nonterminal'][:, self.history + self.n - 1],
                                    dtype=torch.float32, device=self.device)

        return probs, idxs, tree_idxs, states, actions, R, next_states, nonterminals, battery, next_battery, last_action, next_last_action, \
               orientation, next_orientation

    def sample(self, batch_size):
        p_total = self.transitions.total()  # Retrieve sum of all priorities (used to create a normalised probability distribution)
        probs, idxs, tree_idxs, states, actions, returns, next_states, nonterminals, battery, next_battery, \
        last_action, next_last_action, orientation, next_orientation = self._get_samples_from_segments(batch_size,
                                                                                                     p_total)  # Get batch of valid samples
        probs = probs / p_total  # Calculate normalised probabilities
        capacity = self.capacity if self.transitions.full else self.transitions.index
        weights = (capacity * probs) ** -self.priority_weight  # Compute importance-sampling weights w
        weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch
        return tree_idxs, states, actions, returns, next_states, nonterminals, weights, battery, next_battery, \
               last_action, next_last_action, orientation, next_orientation

    def update_priorities(self, idxs, priorities):
        priorities = np.power(priorities, self.priority_exponent)
        self.transitions.update(idxs, priorities)

    # Set up internal state for iterator
    def __iter__(self):
        self.current_idx = 0
        return self

    # Return valid states for validation
    def __next__(self):
        if self.current_idx == self.capacity:
            raise StopIteration
        transitions = self.transitions.data[np.arange(self.current_idx - self.history + 1, self.current_idx + 1)]
        transitions_firsts = transitions['timestep'] == 0
        blank_mask = np.zeros_like(transitions_firsts, dtype=np.bool_)
        for t in reversed(range(self.history - 1)):
            blank_mask[t] = np.logical_or(blank_mask[t + 1],
                                          transitions_firsts[t + 1])  # If future frame has timestep 0
        transitions[blank_mask] = blank_trans
        state = torch.tensor(transitions['state'], dtype=torch.float32,
                             device=self.device)/255  # Agent will turn into batch
        battery = torch.tensor(transitions['battery'][-1], dtype=torch.int32,
                               device=self.device)  # Agent will turn into batch

        l_a = transitions['last_action']
        last_action = torch.tensor(l_a.copy(), dtype=torch.int64,
                                   device=self.device)  # Agent will turn into batch
        o_b = transitions['orientation']
        orientation = torch.tensor(o_b.copy(), dtype=torch.int64,
                                  device=self.device)  # Agent will turn into batch
        self.current_idx += 1
        return state, battery, last_action, orientation

    next = __next__  # Alias __next__ for Python 2 compatibility