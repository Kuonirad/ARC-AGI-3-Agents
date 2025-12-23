import numpy as np
import random
import heapq
from collections import deque, Counter
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v3.0-KEVIN_KULL
    The 'Optimistic' Polymath.

    PHILOSOPHY:
    - Optimistic Physics: Assume everything is walkable until we crash.
    - Wiggle-Calibration: Try all 4 directions before giving up on having a body.
    - Dynamic Identity: Tracking logic that handles color/shape shifts (Keys, Portals).
    """

    # Robust Action Map
    ACT_UP = GameAction.ACTION1
    ACT_DOWN = GameAction.ACTION2
    ACT_LEFT = GameAction.ACTION3
    ACT_RIGHT = GameAction.ACTION4
    ACT_USE = getattr(GameAction, 'ACTION5', GameAction.ACTION5)   # Space
    ACT_CLICK = getattr(GameAction, 'ACTION6', GameAction.ACTION6) # Click

    MAX_ACTIONS = 400

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # --- Knowledge Graph ---
        self.walls = set()        # {(r,c)} - Proven Obstacles
        self.interactions = set() # {(r,c)} - Clicked locations
        self.bad_goals = set()    # {(r,c)} - Unreachable targets

        # --- State Tracking ---
        self.mode = "CALIBRATING"
        self.calibration_moves = [
            self.ACT_RIGHT, self.ACT_DOWN, self.ACT_LEFT, self.ACT_UP
        ]
        self.plan = deque()
        self.last_state = None    # (agent_rc, grid_hash, grid_bytes)
        self.last_grid = None     # Full grid
        self.last_action = None
        self.last_pos = None      # (r,c)
        self.stuck_counter = 0
        self.inventory_hash = 0
        self.agent_color = None   # Dynamic Identity
        self.boot_steps = 0

        # --- Payload Shim ---
        self.action_data = {}

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        self.action_data = {} # Reset
        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET

        try:
            return self._kull_logic(latest)
        except Exception:
            return self._random_move()

    def _kull_logic(self, latest: FrameData) -> GameAction:
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape
        current_hash = np.sum(grid)
        current_bytes = grid.tobytes()

        # 1. LOCATE AGENT (Robust)
        agent_rc = self._locate_agent(latest, grid)

        # 2. DYNAMIC IDENTITY (If lost, find who moved)
        if self.mode == "AVATAR" and self.last_grid is not None and self.last_pos is not None:
            prev_rc = self.last_pos
            prev_bytes = self.last_state[2] if self.last_state else b""

            # If world changed but we are lost/static
            grid_changed = (current_bytes != prev_bytes)

            if grid_changed:
                if agent_rc is None or agent_rc == prev_rc:
                     new_color = self._detect_moving_color(self.last_grid, grid, prev_rc)
                     if new_color is not None:
                         self.agent_color = new_color
                         agent_rc = self._locate_agent(latest, grid) # Retry

        # --- L0: WIGGLE CALIBRATION (Steps 1-4) ---
        if self.mode == "CALIBRATING":
            # Check results of last move
            if self.last_state:
                prev_rc = self.last_pos
                prev_bytes = self.last_state[2]
                grid_changed = (current_bytes != prev_bytes)

                # If we moved or grid changed (physics response)
                if (agent_rc and prev_rc and agent_rc != prev_rc) or grid_changed:
                    self.mode = "AVATAR"
                    self.plan.clear()

            # If still calibrating
            if self.mode == "CALIBRATING":
                step_idx = self.boot_steps
                if step_idx >= len(self.calibration_moves):
                    # Ran out of wiggles -> God Mode
                    self.mode = "SCIENTIST"
                else:
                    # Execute next wiggle
                    move = self.calibration_moves[step_idx]
                    self.boot_steps += 1
                    self._update_state(grid, agent_rc, current_hash, current_bytes)
                    self.last_action = move
                    return move

        # --- L1: PHYSICS UPDATE (Avatar) ---
        # Inventory Change -> Doors Open -> Clear Walls
        if abs(current_hash - self.inventory_hash) > 0:
            self.walls.clear()
            self.plan.clear()
            self.bad_goals.clear()
        self.inventory_hash = current_hash

        # Collision Learning
        if self.mode == "AVATAR" and self.last_pos and self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
            # If we didn't move, and grid didn't change (no push)
            grid_same = (current_bytes == self.last_state[2])

            if agent_rc == self.last_pos and grid_same:
                self.stuck_counter += 1
                if self.stuck_counter > 0:
                    tr, tc = self._get_target(self.last_pos, self.last_action)
                    self.walls.add((tr, tc))
                    self.plan.clear()
            else:
                self.stuck_counter = 0

        # Update State
        self._update_state(grid, agent_rc, current_hash, current_bytes)

        # Decide
        action = self._decide_strategy(grid, agent_rc, rows, cols)
        self.last_action = action
        return action

    def _update_state(self, grid, rc, h, b):
        self.last_grid = grid.copy()
        self.last_state = (rc, h, b)
        self.last_pos = rc

    def _decide_strategy(self, grid, agent_rc, rows, cols):
        if self.plan: return self.plan.popleft()

        if self.mode == "AVATAR":
            return self._strategy_avatar(grid, agent_rc, rows, cols)
        else:
            return self._strategy_scientist(grid, rows, cols)

    def _strategy_avatar(self, grid, agent_rc, rows, cols):
        """Optimistic A* Navigation"""
        if not agent_rc: return self._random_move()

        # 1. Identify Goals (Rare objects)
        targets = self._scan_targets(grid, agent_rc)

        # 2. Pathfind (Optimistic)
        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue

            # OPTIMISTIC: Assume everything is walkable unless in self.walls
            path = self._astar(grid, agent_rc, t_rc, rows, cols)
            if path:
                self.plan.extend(path)
                return self.plan.popleft()
            else:
                self.bad_goals.add(t_rc)

        # 3. Frontier (Explore unknown)
        frontier = self._find_frontier(grid, agent_rc, rows, cols)
        if frontier:
            self.plan.extend(frontier)
            return self.plan.popleft()

        return self._random_move()

    def _strategy_scientist(self, grid, rows, cols):
        """Pattern Extension + Clicking"""
        # 1. Line Extension (ft09)
        ext = self._find_vector_extension(grid, rows, cols)
        if ext: return self._execute_click(ext)

        # 2. Rare Object Probe (vc33)
        targets = self._scan_targets(grid, (-1,-1)) # Scan all
        for _, t_rc in targets:
            if t_rc not in self.interactions:
                return self._execute_click(t_rc)

        # 3. Random
        r = random.randint(0, rows-1)
        c = random.randint(0, cols-1)
        return self._execute_click((r, c))

    def _astar(self, grid, start, end, rows, cols):
        """Optimistic A* (Manhattan)"""
        pq = [(0, 0, start, [])]
        best_g = {start: 0}

        while pq:
            f, steps, curr, path = heapq.heappop(pq)
            if curr == end: return path
            if steps > 500: break

            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # OPTIMISM: If not a known wall, we try it.
                    if (nr, nc) not in self.walls:
                        new_g = steps + 1
                        if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = new_g
                            h = abs(nr - end[0]) + abs(nc - end[1])
                            heapq.heappush(pq, (new_g + h, new_g, (nr, nc), path + [act]))
        return None

    def _find_vector_extension(self, grid, rows, cols):
        """Finds line patterns and clicks the next empty spot."""
        non_zeros = np.argwhere(grid != 0)
        if len(non_zeros) > 80: return None

        by_color = {}
        for r, c in non_zeros:
            val = grid[r, c]
            if val not in by_color: by_color[val] = []
            by_color[val].append((r, c))

        for color, coords in by_color.items():
            if len(coords) < 2: continue
            # Check pairwise
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    r1, c1 = coords[i]
                    r2, c2 = coords[j]
                    dr, dc = r2 - r1, c2 - c1

                    # Must be adjacent or close (step 1 or 2)
                    if abs(dr) > 2 or abs(dc) > 2: continue

                    next_r, next_c = r2 + dr, c2 + dc
                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        if grid[next_r, next_c] == 0 and (next_r, next_c) not in self.interactions:
                            return (next_r, next_c)
        return None

    def _execute_click(self, rc):
        r, c = rc
        self.interactions.add((r, c))
        payload = {'x': int(c), 'y': int(r)}
        self.action_data = payload
        try:
            self.ACT_CLICK.set_data(payload)
        except:
            pass
        return self.ACT_CLICK

    def _locate_agent(self, latest, grid):
        # 1. API
        if hasattr(latest, 'agent_pos') and latest.agent_pos and latest.agent_pos != (-1, -1):
            return (latest.agent_pos[1], latest.agent_pos[0])
        # 2. Dynamic Color
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0: return tuple(coords[0])
        # 3. Heuristic (Allow 0 if rare)
        unique, counts = np.unique(grid, return_counts=True)
        # Sort by rarity
        indices = np.argsort(counts)
        sorted_colors = unique[indices]
        sorted_counts = counts[indices]

        for color, count in zip(sorted_colors, sorted_counts):
            if count < 50: # Assume agent is small
                coords = np.argwhere(grid == color)
                if len(coords) > 0: return tuple(coords[0])
        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 50) & (unique != 0)]
        targets = []
        matches = np.argwhere(np.isin(grid, rare))
        for r, c in matches:
            if (r, c) != start_rc:
                d = abs(r - start_rc[0]) + abs(c - start_rc[1])
                targets.append((d, (r, c)))
        targets.sort()
        return targets

    def _detect_moving_color(self, prev_grid, curr_grid, last_pos):
        if prev_grid.shape != curr_grid.shape: return None
        if not last_pos: return None
        r, c = last_pos
        diff = prev_grid != curr_grid
        if not np.any(diff): return None

        # Look around last position for new values
        neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
        rows, cols = curr_grid.shape
        candidates = []
        for nr, nc in neighbors:
             if 0 <= nr < rows and 0 <= nc < cols:
                 val = curr_grid[nr, nc]
                 if val != 0 and val != prev_grid[nr, nc]:
                     candidates.append(val)
        if candidates: return Counter(candidates).most_common(1)[0][0]
        return None

    def _find_frontier(self, grid, start, rows, cols):
        """BFS to unknown"""
        q = deque([(start, [])])
        seen = {start}
        steps = 0
        while q:
            steps += 1
            if steps > 200: break
            curr, path = q.popleft()
            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]
            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in self.walls: continue
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path+[act]))
            if len(path) > 8: return path
        return None

    def _get_target(self, rc, action):
        r, c = rc
        if action == self.ACT_UP: return r-1, c
        if action == self.ACT_DOWN: return r+1, c
        if action == self.ACT_LEFT: return r, c-1
        if action == self.ACT_RIGHT: return r, c+1
        return r, c

    def _random_move(self):
        return random.choice([self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT])
