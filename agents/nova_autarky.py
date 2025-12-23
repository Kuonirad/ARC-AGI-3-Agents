import random
from collections import deque

import numpy as np

from .agent import Agent
from .structs import FrameData, GameAction, GameState


class NovaAutarkyAgent(Agent):
    """
    NOVA-M-COP v1.6-AUTARKY
    Sovereign Agent using Test-Time Training (TTT) & Active Inference.
    - Learns Walls via Collision.
    - Detects Goals via Rarity.
    - Adapts via State Hashing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = deque()
        self.knowledge = {
            "walls": set(),  # Coordinates of discovered walls
            "bad_goals": set(),  # Unreachable targets
        }
        self.last_pos = None
        self.last_action = None
        self.state_hash = None  # Tracks inventory/shape changes
        self.stuck_count = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        """The Autarkic O.O.D.A Loop"""
        if latest.state == GameState.NOT_PLAYED:
            return GameAction.RESET

        if not latest.frame:
            return GameAction.RESET

        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape

        agent_pos = self._get_agent_pos(grid)

        # 1. PERCEPT & LEARN (L2 Ephemeral Memory)
        # Did our last action fail? (Collision Detection)
        if self.last_pos and self.last_action:
            if self.last_pos == agent_pos:
                self.stuck_count += 1
                # Calculate where we tried to go
                dr, dc = self._action_to_delta(self.last_action)
                blocked_r, blocked_c = self.last_pos[0] + dr, self.last_pos[1] + dc

                # LEARN: This tile is a wall.
                self.knowledge["walls"].add((blocked_r, blocked_c))

                # If stuck, abort current plan and re-think
                if self.stuck_count > 0:
                    self.plan.clear()
            else:
                self.stuck_count = 0

        self.last_pos = agent_pos

        # 2. STATE CHECK (L1 Retina)
        # Did I change shape/color? (e.g., Picked up key)
        curr_hash = grid[agent_pos]
        if self.state_hash is not None and curr_hash != self.state_hash:
            # Major event! Clear plan to leverage new capabilities (e.g., unlocking door)
            self.plan.clear()
            self.knowledge["bad_goals"].clear()  # Retry previously failed goals
        self.state_hash = curr_hash

        # 3. EXECUTE EXISTING PLAN
        if self.plan:
            action = self.plan.popleft()
            self.last_action = action
            return action

        # 4. ORIENT (L3 Goal Inference)
        # Scan for objects. Filter out known walls and background (0).
        # Heuristic: Go to the nearest 'Unique' object.
        targets = self._scan_for_targets(grid, agent_pos)

        best_path = None
        for target_pos in targets:
            if target_pos in self.knowledge["bad_goals"]:
                continue

            # Try to pathfind
            path = self._bfs(grid, agent_pos, target_pos)
            if path:
                best_path = path
                break
            else:
                # Mark as unreachable for now
                self.knowledge["bad_goals"].add(target_pos)

        # 5. ACT
        if best_path:
            self.plan.extend(best_path)
            action = self.plan.popleft()
            self.last_action = action
            return action

        # 6. FALLBACK (Explorer)
        # Random walk if no targets or trapped
        action = random.choice(
            [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
            ]
        )
        self.last_action = action
        return action

    def _get_agent_pos(self, grid):
        """
        Heuristic to find the agent position.
        In ls20 and similar games, the agent is usually a unique colored pixel.
        """
        rows, cols = grid.shape
        # Priority search for single pixels of specific colors
        # Adjust these colors based on game knowledge if needed.
        # For now, just look for ANY single pixel color that isn't 0 (background).

        unique, counts = np.unique(grid, return_counts=True)
        # Filter for counts == 1 and value != 0
        potential_agents = unique[(counts == 1) & (unique != 0)]

        if len(potential_agents) > 0:
            # Pick the first one? Or prefer certain colors?
            # Let's pick the one that matches our previous hash if valid?
            # Or just the first one.
            agent_color = potential_agents[0]
            coords = np.argwhere(grid == agent_color)
            if len(coords) > 0:
                return tuple(coords[0])

        # If no single pixel found, fallback to searching for specific common agent colors
        # e.g. Blue (1), Red (2), Cyan (8)
        for color in [1, 2, 8]:
            coords = np.argwhere(grid == color)
            if len(coords) == 1:
                return tuple(coords[0])

        return (0, 0)

    def _scan_for_targets(self, grid, agent_pos):
        """Find rare objects, sorted by distance."""
        unique, counts = np.unique(grid, return_counts=True)
        # Rare objects are < 15 pixels (Keys, Doors, Switches)
        # Background is usually 0.
        rare_colors = unique[(counts < 15) & (unique != 0)]

        candidates = []
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] in rare_colors and (r, c) != agent_pos:
                    dist = abs(r - agent_pos[0]) + abs(c - agent_pos[1])
                    candidates.append((dist, (r, c)))

        candidates.sort()  # Closest first
        return [c[1] for c in candidates]

    def _bfs(self, grid, start, end):
        """Deterministic Pathfinding respecting Learned Walls."""
        q = deque([(start, [])])
        seen = {start}
        rows, cols = grid.shape

        while q:
            curr, path = q.popleft()
            if curr == end:
                return path

            if len(path) > 50:
                continue  # Limit depth for speed

            r, c = curr
            moves = [
                ((-1, 0), GameAction.ACTION1),
                ((1, 0), GameAction.ACTION2),
                ((0, -1), GameAction.ACTION3),
                ((0, 1), GameAction.ACTION4),
            ]

            for (dr, dc), act in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Check Learned Walls
                    if (nr, nc) in self.knowledge["walls"]:
                        continue

                    # Heuristic: Avoid stepping on other objects unless it's the target
                    # (Prevents getting stuck on non-goal clutter)
                    if (nr, nc) != end and grid[nr, nc] != 0:
                        # Treat unknown non-0 as potential obstacle until proven otherwise?
                        # For LS20, we treat them as walkable to interact.
                        pass

                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path + [act]))
        return None

    def _action_to_delta(self, action):
        if action == GameAction.ACTION1:
            return (-1, 0)
        if action == GameAction.ACTION2:
            return (1, 0)
        if action == GameAction.ACTION3:
            return (0, -1)
        if action == GameAction.ACTION4:
            return (0, 1)
        return (0, 0)
