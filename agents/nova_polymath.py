import heapq
import random
from collections import deque

import numpy as np

from .agent import Agent
from .structs import FrameData, GameAction, GameState


class NovaPolymathAgent(Agent):
    """
    NOVA-M-COP v1.9-POLYMATH
    The 'Rigorous' Solver.
    1. DYNAMIC PHYSICS: Walls are state-dependent. Key pickup CLEARS wall memory.
    2. MODE SWITCHING: Detects Avatar (ls20) vs God/Cursor (vc33).
    3. ACTION HYGIENE: Uses ACTION6 with 'action_data' payload for clicks.
    4. COORDINATE RIGOR: Strict Row(y)/Col(x) handling.
    """

    # Standard Action Set
    ACT_UP = GameAction.ACTION1
    ACT_DOWN = GameAction.ACTION2
    ACT_LEFT = GameAction.ACTION3
    ACT_RIGHT = GameAction.ACTION4
    ACT_USE = getattr(GameAction, "ACTION5", GameAction.ACTION5)  # Space/Interact
    ACT_CLICK = getattr(GameAction, "ACTION6", GameAction.ACTION6)  # Click

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # KNOWLEDGE BASE
        self.walls = set()  # {(r, c)}
        self.visited = set()  # {(r, c)}
        self.interactions = set()  # {(r, c)}

        # STATE TRACKING
        self.plan = deque()
        self.last_state = None  # (agent_rc, grid_hash)
        self.last_action = None
        self.stuck_counter = 0
        self.inventory_hash = 0  # Sum of pixel values (detects pickup/change)

        # MODES
        self.mode = "ANALYZING"  # ANALYZING -> NAVIGATOR or SCIENTIST
        self.boot_steps = 0

        # Payload for coordinate actions
        self.action_data = {}

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        """The Polymathic Loop: Percept -> Update Model -> Plan -> Act"""
        if latest.state == GameState.NOT_PLAYED:
            return GameAction.RESET
        if not latest.frame:
            return GameAction.RESET

        try:
            return self._rigorous_choose(latest)
        except Exception:
            return self._random_move()

    def _rigorous_choose(self, latest: FrameData) -> GameAction:
        # 1. RIGOROUS PERCEPTION
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape
        agent_rc = self._locate_agent(latest, grid)

        # Hash the world state (excluding agent pos) to detect inventory/env changes
        current_hash = np.sum(grid)

        # 2. MODE DETECTION (Boot Phase)
        if self.mode == "ANALYZING":
            self.boot_steps += 1
            if agent_rc is not None:
                self.mode = "NAVIGATOR"  # We have a body
            elif self.boot_steps > 1:
                self.mode = "SCIENTIST"  # No body found, assuming God Mode

        # 3. DYNAMIC PHYSICS UPDATE (The Fix for 'Permanent Walls')
        # If the world mass changed (e.g. Key disappeared), our physics model is outdated.
        # FLUSH WALLS. We might be able to pass through doors now.
        if abs(current_hash - self.inventory_hash) > 0:
            self.walls.clear()  # TABULA RASA
            self.plan.clear()
        self.inventory_hash = current_hash

        # 4. COLLISION LEARNING (Navigator Only)
        if self.mode == "NAVIGATOR" and self.last_state:
            prev_rc, prev_hash = self.last_state
            if self.last_action in [
                self.ACT_UP,
                self.ACT_DOWN,
                self.ACT_LEFT,
                self.ACT_RIGHT,
            ]:
                if prev_rc == agent_rc and agent_rc is not None:
                    # We didn't move. But did the world change? (e.g. Pushed a box)
                    if prev_hash == current_hash:
                        # We hit a solid wall.
                        self.stuck_counter += 1
                        if self.stuck_counter > 0:
                            target_rc = self._get_target(prev_rc, self.last_action)
                            self.walls.add(target_rc)
                            self.plan.clear()  # Re-plan required
                    else:
                        # Interaction occurred (e.g. Push). Do not mark as wall.
                        self.stuck_counter = 0
                else:
                    self.stuck_counter = 0

        self.last_state = (agent_rc, current_hash)

        # 5. EXECUTE PLAN
        if self.plan:
            action = self.plan.popleft()
            self.last_action = action
            return action

        # 6. HIGH-LEVEL STRATEGY
        if self.mode == "SCIENTIST":
            action = self._strategy_scientist(grid, rows, cols)
        else:
            action = self._strategy_navigator(grid, agent_rc, rows, cols)

        self.last_action = action
        return action

    def _strategy_navigator(self, grid, start_rc, rows, cols):
        """Logic for ls20: Find Rare Objects -> A* Path"""
        if not start_rc:
            return self._random_move()

        # HEURISTIC: Go to nearest object we haven't 'consumed'
        targets = self._scan_targets(grid, start_rc)

        for dist, t_rc in targets:
            if t_rc in self.walls:
                continue

            # A* Search
            path = self._astar(grid, start_rc, t_rc, rows, cols)
            if path:
                self.plan.extend(path)
                return self.plan.popleft()

        # FRONTIER EXPLORATION: If no objects reachable, go to unknown space
        frontier_path = self._find_frontier(grid, start_rc, rows, cols)
        if frontier_path:
            self.plan.extend(frontier_path)
            return self.plan.popleft()

        return self._random_move()

    def _strategy_scientist(self, grid, rows, cols):
        """Logic for vc33: Click Control"""
        # Strategy: Click on non-background pixels we haven't clicked yet.
        unique, counts = np.unique(grid, return_counts=True)
        # Sort objects by rarity
        rare_indices = np.argsort(counts)
        sorted_vals = unique[rare_indices]

        target_rc = None
        for val in sorted_vals:
            if val == 0:
                continue  # Skip black
            matches = np.argwhere(grid == val)
            for r, c in matches:
                if (r, c) not in self.interactions:
                    target_rc = (r, c)
                    break
            if target_rc:
                break

        if target_rc:
            r, c = target_rc
            self.interactions.add((r, c))
            # ACTION6 Payload: Set coordinates for the framework
            # Note: API uses (x, y), we have (r, c)
            # Correctly apply data to the Enum Member (temporarily or permanently?)
            # GameAction is Enum. 'action_data' is updated on the Enum Member instance.
            # This is globally shared state but given single threaded per game, fine.
            self.ACT_CLICK.set_data({"x": int(c), "y": int(r)})
            return self.ACT_CLICK

        # Fallback: Random click
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        self.ACT_CLICK.set_data({"x": c, "y": r})
        return self.ACT_CLICK

    def _astar(self, grid, start, end, rows, cols):
        """A* Pathfinding (Manhattan Distance)"""
        pq = [(0, 0, start, [])]
        best_g = {start: 0}

        while pq:
            f, steps, curr, path = heapq.heappop(pq)
            if curr == end:
                return path
            if steps > 1000:
                break

            r, c = curr
            moves = [
                ((-1, 0), self.ACT_UP),
                ((1, 0), self.ACT_DOWN),
                ((0, -1), self.ACT_LEFT),
                ((0, 1), self.ACT_RIGHT),
            ]

            for (dr, dc), act in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # LOGIC: Walkable if 0 OR it is the Goal
                    # (Walking onto the Key/Door is necessary)
                    is_walkable = (grid[nr, nc] == 0) or ((nr, nc) == end)
                    if (nr, nc) in self.walls:
                        is_walkable = False

                    if is_walkable:
                        new_g = steps + 1
                        if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = new_g
                            h = abs(nr - end[0]) + abs(nc - end[1])
                            heapq.heappush(
                                pq, (new_g + h, new_g, (nr, nc), path + [act])
                            )
        return None

    def _find_frontier(self, grid, start, rows, cols):
        """BFS to nearest reachable tile"""
        q = deque([(start, [])])
        seen = {start}
        steps = 0
        while q:
            steps += 1
            if steps > 500:
                break
            curr, path = q.popleft()

            r, c = curr
            moves = [
                ((-1, 0), self.ACT_UP),
                ((1, 0), self.ACT_DOWN),
                ((0, -1), self.ACT_LEFT),
                ((0, 1), self.ACT_RIGHT),
            ]

            for (dr, dc), act in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in self.walls:
                        continue
                    if grid[nr, nc] != 0:
                        continue

                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path + [act]))

            if len(path) > 5 and random.random() < 0.1:
                return path
        return None

    def _locate_agent(self, latest, grid):
        """Robust Agent Locator (Row, Col)"""
        # 1. Trust API (Convert x,y -> r,c)
        if (
            hasattr(latest, "agent_pos")
            and latest.agent_pos
            and latest.agent_pos != (-1, -1)
        ):
            return (latest.agent_pos[1], latest.agent_pos[0])

        # 2. Visual Scan (Colors 1, 2, 8 are common agents)
        for color in [1, 2, 8, 3]:
            matches = np.argwhere(grid == color)
            if len(matches) == 1:
                return tuple(matches[0])

        # 3. Rarest single pixel
        unique, counts = np.unique(grid, return_counts=True)
        sorted_indices = np.argsort(counts)
        sorted_colors = unique[sorted_indices]
        for color in sorted_colors:
            if color == 0:
                continue
            if counts[np.where(unique == color)[0][0]] < 50:
                coords = np.argwhere(grid == color)
                if len(coords) > 0:
                    return tuple(coords[0])

        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare_colors = unique[(counts < 25) & (unique != 0)]
        targets = []
        matches = np.argwhere(np.isin(grid, rare_colors))
        for r, c in matches:
            if (r, c) != start_rc:
                d = abs(r - start_rc[0]) + abs(c - start_rc[1])
                targets.append((d, (r, c)))
        targets.sort()
        return targets

    def _get_target(self, rc, action):
        r, c = rc
        if action == self.ACT_UP:
            return r - 1, c
        if action == self.ACT_DOWN:
            return r + 1, c
        if action == self.ACT_LEFT:
            return r, c - 1
        if action == self.ACT_RIGHT:
            return r, c + 1
        return r, c

    def _random_move(self):
        return random.choice(
            [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]
        )
