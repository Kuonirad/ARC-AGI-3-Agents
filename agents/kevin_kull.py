import numpy as np
import random
import heapq
from collections import deque, Counter
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class NovaSingularityAgent(Agent):
    """
    NOVA-M-COP v2.0-SINGULARITY
    The 'Polymathic' Generalist.

    CAPABILITIES:
    1. AVATAR: Solves Navigation (ls20) via Dynamic A* & State-Dependent Physics.
    2. SCIENTIST: Solves Orchestration (vc33) via Rarity probing.
    3. BUILDER: Solves Patterning (ft09) via Vector Extrapolation (Line Extension).

    FIXES:
    - Coordinate Hygiene: Strict (row, col) management.
    - Action Payload: Robustly attaches (x,y) data to clicks via multiple shims.
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
        self.walls = set()        # {(r,c)} - Obstacles
        self.interactions = set() # {(r,c)} - Clicked locations
        self.bad_goals = set()    # {(r,c)} - Unreachable/Useless targets

        # --- State Tracking ---
        self.mode = "CALIBRATING"
        self.plan = deque()
        self.last_state = None    # (agent_rc, grid_hash)
        self.last_action = None
        self.stuck_counter = 0
        self.inventory_hash = 0
        self.boot_steps = 0
        self.agent_color = None

        # --- Payload Shim ---
        self.action_data = {}

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        # Reset payload every tick to prevent leakage
        self.action_data = {}

        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET

        try:
            return self._polymathic_choose(latest)
        except Exception as e:
            # Fallback for robustness (never freeze)
            return self._random_move()

    def _polymathic_choose(self, latest: FrameData) -> GameAction:
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape
        agent_rc = self._locate_agent(latest, grid)
        current_hash = np.sum(grid)

        # --- L0: PHYSICS CALIBRATION ---
        if self.mode == "CALIBRATING":
            self.boot_steps += 1

            # 1. Try to Move (Generates delta)
            if self.boot_steps == 1:
                move = self._find_open_move(grid, agent_rc)
                self.last_state = (agent_rc, current_hash, grid.tobytes())
                self.last_action = move
                return move

            # 2. Analyze Result
            prev_rc, prev_hash, prev_bytes = self.last_state

            # Did we move? (Body exists)
            pos_changed = (agent_rc and prev_rc and agent_rc != prev_rc)
            # Use raw bytes for grid change, as sum() is invariant to movement
            grid_changed = (grid.tobytes() != prev_bytes)

            # Dynamic Color Logic: If grid changed but agent didn't, we might be tracking wrong object
            if grid_changed and not pos_changed:
                 last_grid = np.frombuffer(prev_bytes, dtype=grid.dtype).reshape(grid.shape)
                 diff = grid != last_grid
                 changed_values = grid[diff]
                 candidates = [v for v in changed_values if v != 0]
                 if candidates:
                     from collections import Counter
                     most_common = Counter(candidates).most_common(1)
                     if most_common:
                         self.agent_color = most_common[0][0]
                         agent_rc = self._locate_agent(latest, grid)
                         pos_changed = (agent_rc != prev_rc)

            if pos_changed:
                self.mode = "AVATAR"
            elif grid_changed:
                # Grid changed -> Physics exist -> Avatar/Builder
                self.mode = "AVATAR"
            else:
                # Static. Likely a "Click to solve" puzzle (ft09/vc33)
                self.mode = "SCIENTIST"

        # --- L1: DYNAMIC PHYSICS (Avatar Mode) ---
        # If inventory changed (Key pickup), clear walls (Doors might open)
        if abs(current_hash - self.inventory_hash) > 0:
            self.walls.clear()
            self.plan.clear()
            self.bad_goals.clear()
        self.inventory_hash = current_hash

        # Collision Detection
        if self.mode == "AVATAR" and self.last_state:
            prev_rc, prev_hash = self.last_state
            if self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
                if prev_rc == agent_rc and agent_rc is not None and prev_hash == current_hash:
                    self.stuck_counter += 1
                    if self.stuck_counter > 0:
                        tr, tc = self._get_target(prev_rc, self.last_action)
                        self.walls.add((tr, tc))
                        self.plan.clear()
                else:
                    self.stuck_counter = 0

        self.last_state = (agent_rc, current_hash, grid.tobytes())

        # --- EXECUTION ---
        if self.plan:
            return self.plan.popleft()

        if self.mode == "AVATAR":
            return self._strategy_avatar(grid, agent_rc, rows, cols)
        else:
            # Hybrid Scientist/Builder Strategy
            return self._strategy_singularity(grid, rows, cols)

    def _strategy_avatar(self, grid, agent_rc, rows, cols):
        """Navigation Logic (A*)"""
        if not agent_rc: return self._random_move()

        # 1. Goal Inference (Rare objects)
        targets = self._scan_targets(grid, agent_rc)

        # 2. Pathfind
        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue

            path = self._astar(grid, agent_rc, t_rc, rows, cols)
            if path:
                self.plan.extend(path)
                return self.plan.popleft()
            else:
                self.bad_goals.add(t_rc)

        # 3. Frontier Exploration (If no goals)
        frontier = self._find_frontier(grid, agent_rc, rows, cols)
        if frontier:
            self.plan.extend(frontier)
            return self.plan.popleft()

        return self._random_move()

    def _strategy_singularity(self, grid, rows, cols):
        """
        Combined Scientist (Click) + Builder (Pattern Extension).
        Prioritizes:
        1. Extending Lines (ft09)
        2. Clicking Rare Objects (vc33)
        3. Random Probing
        """
        # --- PHASE 1: BUILDER (Vector Extrapolation) ---
        # Look for colored lines and click the empty spot at the end.
        extension_target = self._find_vector_extension(grid, rows, cols)
        if extension_target:
            return self._execute_click(extension_target)

        # --- PHASE 2: SCIENTIST (Rarity Probe) ---
        # Click non-background objects we haven't touched.
        unique, counts = np.unique(grid, return_counts=True)
        rare_indices = np.argsort(counts)
        sorted_vals = unique[rare_indices]

        for val in sorted_vals:
            if val == 0: continue
            matches = np.argwhere(grid == val)
            for r, c in matches:
                if (r, c) not in self.interactions:
                    return self._execute_click((r, c))

        # --- PHASE 3: ENTROPY (Random) ---
        r = random.randint(0, rows-1)
        c = random.randint(0, cols-1)
        return self._execute_click((r, c))

    def _find_vector_extension(self, grid, rows, cols):
        """
        Detects 2+ pixels of same color forming a line.
        Returns coordinate of the NEXT pixel in that line if it's 0 (empty).
        """
        # Scan for colored pixels
        non_zeros = np.argwhere(grid != 0)
        if len(non_zeros) > 60: return None # Too noisy/complex

        # Group by color
        by_color = {}
        for r, c in non_zeros:
            val = grid[r, c]
            if val not in by_color: by_color[val] = []
            by_color[val].append((r, c))

        for color, coords in by_color.items():
            if len(coords) < 2: continue

            # Check every pair for a vector
            # Limit checks for speed (nearest neighbors)
            peers = coords[:12]
            for i in range(len(peers)):
                for j in range(i+1, len(peers)):
                    r1, c1 = peers[i]
                    r2, c2 = peers[j]

                    # Vector (dr, dc)
                    dr = r2 - r1
                    dc = c2 - c1

                    # Simplify vector (get unit step or exact step)
                    # For pattern completion, we usually want exact step repetition
                    step_r = dr
                    step_c = dc

                    # Project forward from (r2, c2)
                    next_r, next_c = r2 + step_r, c2 + step_c

                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        # If empty and not clicked -> Candidate
                        if grid[next_r, next_c] == 0 and (next_r, next_c) not in self.interactions:
                            return (next_r, next_c)

                    # Project backward from (r1, c1)
                    prev_r, prev_c = r1 - step_r, c1 - step_c
                    if 0 <= prev_r < rows and 0 <= prev_c < cols:
                         if grid[prev_r, prev_c] == 0 and (prev_r, prev_c) not in self.interactions:
                            return (prev_r, prev_c)
        return None

    def _execute_click(self, rc):
        r, c = rc
        self.interactions.add((r, c))

        # SHIM: Attach data to the Enum Member
        payload = {'x': int(c), 'y': int(r)}

        try:
            self.ACT_CLICK.set_data(payload)
        except:
            pass

        return self.ACT_CLICK

    def _astar(self, grid, start, end, rows, cols):
        """A* (Manhattan)"""
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
                    # Walkable if Air (0) OR Goal (Walking onto key/door/button is good)
                    is_walkable = (grid[nr, nc] == 0) or ((nr, nc) == end)
                    if (nr, nc) in self.walls: is_walkable = False

                    if is_walkable:
                        new_g = steps + 1
                        if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = new_g
                            h = abs(nr - end[0]) + abs(nc - end[1])
                            heapq.heappush(pq, (new_g + h, new_g, (nr, nc), path + [act]))
        return None

    def _find_frontier(self, grid, start, rows, cols):
        """BFS to unknown space"""
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
                    if (nr, nc) in self.walls or grid[nr, nc] != 0: continue
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path+[act]))

            # If deep enough, consider it a frontier
            if len(path) > 8: return path
        return None

    def _locate_agent(self, latest, grid):
        """Robust Location"""
        # 0. Learned Color
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0:
                 return tuple(coords[0])

        if hasattr(latest, 'agent_pos') and latest.agent_pos and latest.agent_pos != (-1, -1):
            return (latest.agent_pos[1], latest.agent_pos[0])

        # Visual Scan (Colors 1, 2, 8 are common agents)
        for color in [1, 2, 8, 3]:
            matches = np.argwhere(grid == color)
            if len(matches) == 1: return tuple(matches[0])

        unique, counts = np.unique(grid, return_counts=True)
        sorted_indices = np.argsort(counts)
        sorted_colors = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]

        for color, count in zip(sorted_colors, sorted_counts):
            if color == 0 and count > 100: continue
            if 0 < count < 50:
                coords = np.argwhere(grid == color)
                if len(coords) > 0: return tuple(coords[0])
        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 40) & (unique != 0)]
        targets = []
        matches = np.argwhere(np.isin(grid, rare))
        for r, c in matches:
            if (r, c) != start_rc:
                d = abs(r - start_rc[0]) + abs(c - start_rc[1])
                targets.append((d, (r, c)))
        targets.sort()
        return targets

    def _get_target(self, rc, action):
        r, c = rc
        if action == self.ACT_UP: return r-1, c
        if action == self.ACT_DOWN: return r+1, c
        if action == self.ACT_LEFT: return r, c-1
        if action == self.ACT_RIGHT: return r, c+1
        return r, c

    def _find_open_move(self, grid, rc):
        if not rc: return self.ACT_RIGHT
        r, c = rc
        h, w = grid.shape
        # Prioritize Right, then others
        if c < w-1 and grid[r, c+1] == 0: return self.ACT_RIGHT
        if r < h-1 and grid[r+1, c] == 0: return self.ACT_DOWN
        if c > 0 and grid[r, c-1] == 0: return self.ACT_LEFT
        if r > 0 and grid[r-1, c] == 0: return self.ACT_UP
        return self.ACT_RIGHT

    def _random_move(self):
        return random.choice([self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT])
