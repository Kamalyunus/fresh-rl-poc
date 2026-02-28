"""
SumTree data structure for O(log n) proportional sampling in Prioritized Experience Replay.

A binary tree stored as a flat numpy array where:
- Leaf nodes hold transition priorities
- Internal nodes hold the sum of their children
- A parallel min-tree tracks the minimum priority for O(1) lookup

References:
    Schaul et al., "Prioritized Experience Replay", 2015
"""

import numpy as np


class SumTree:
    """
    Binary sum-tree with parallel min-tree for prioritized sampling.

    The tree has `capacity` leaves. Internal nodes store the sum (or min)
    of their children, enabling O(log n) proportional sampling and O(1)
    total/min queries.

    Storage layout (capacity=4):
        Index:  0    1    2    3    4    5    6
                     [root]
                   /        \\
                [1]          [2]
               /   \\        /   \\
             [3]   [4]    [5]   [6]

        Leaves are indices [capacity-1, 2*capacity-2].
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.min_tree = np.full(2 * capacity - 1, np.inf, dtype=np.float64)
        self.data = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def _propagate_sum(self, idx: int):
        """Update sum-tree from leaf to root."""
        parent = (idx - 1) // 2
        while parent >= 0:
            left = 2 * parent + 1
            right = 2 * parent + 2
            self.tree[parent] = self.tree[left] + self.tree[right]
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def _propagate_min(self, idx: int):
        """Update min-tree from leaf to root."""
        parent = (idx - 1) // 2
        while parent >= 0:
            left = 2 * parent + 1
            right = 2 * parent + 2
            self.min_tree[parent] = min(self.min_tree[left], self.min_tree[right])
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def add(self, priority: float, data):
        """Add a new transition with given priority, overwriting oldest if full."""
        tree_idx = self.write_idx + self.capacity - 1

        self.data[self.write_idx] = data
        self.tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority
        self._propagate_sum(tree_idx)
        self._propagate_min(tree_idx)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, tree_idx: int, priority: float):
        """Update priority at a given tree index and propagate."""
        self.tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority
        self._propagate_sum(tree_idx)
        self._propagate_min(tree_idx)

    def get(self, cumulative_sum: float):
        """
        Find the leaf whose cumulative sum bracket contains the query value.

        Returns (tree_idx, priority, data).
        """
        idx = 0  # start at root
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2

            if left >= len(self.tree):
                # Reached a leaf
                break

            if cumulative_sum <= self.tree[left]:
                idx = left
            else:
                cumulative_sum -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    def total(self) -> float:
        """Total sum of all priorities (root node)."""
        return self.tree[0]

    def min_priority(self) -> float:
        """Minimum priority across all leaves (root of min-tree)."""
        if self.size == 0:
            return 0.0
        return self.min_tree[0]
