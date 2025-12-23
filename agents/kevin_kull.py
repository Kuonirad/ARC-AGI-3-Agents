import numpy as np
import random
import heapq
from collections import deque, Counter, defaultdict
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v4.0-OMEGA (KEVIN_KULL)
    The 'Recursive Synthesis' Agent.

    HIERARCHY:
    1. EXECUTIVE (Metacognition): Resolves conflicts via Recursive Critique (Reachability Check).
    2. PERSPECTIVES (The Parliament):
       - NEWTON (Nav): Soft A* with Dynamic Obstacle Costing.
       - EUCLID (Pattern): Vector Extrapolation + Symmetry Mirroring.
       - SKINNER (Causal): Interaction weighted by History of Effect.
    3. SUBSTRATE (Physics):
       - Dynamic Identity (Shape-Shifting).
       - Autopoietic Calibration (Wiggle-Test).

    CRITICAL FIX v4.0: Sanitized Bidding (Prevents Payload Race Conditions).
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
        self.walls = set()        # Hard constraints
        self.bad_goals = set()    # Unreachable
        self.causal_history = defaultdict(float) # ObjectVal -> ImpactScore
        self.interactions = set() # Locations clicked

        # --- State ---
        self.mode = "CALIBRATING"
        self.boot_steps = 0
        self.last_state = None    # (rc, hash)
        self.last_grid = None
        self.last_action = None
        self.last_pos = None
        self.stuck_counter = 0
        self.inventory_hash = 0
        self.agent_color = None
        self.action_data = {}

        # --- Calibration ---
        self.calibration_moves = [
            self.ACT_RIGHT, self.ACT_DOWN, self.ACT_LEFT, self.ACT_UP
        ]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        self.action_data = {} # Clear payload shim
        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET
        try:
            return self._omega_executive(latest)
        except Exception:
            return self._perspective_chaos()

    def _omega_executive(self, latest: FrameData) -> GameAction:
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape
        current_hash = np.sum(grid)

        # 1. PERCEPTION & PHYSICS UPDATE
        agent_rc = self._locate_agent(latest, grid)
        self._update_physics(grid, agent_rc, current_hash)

        # 2. CALIBRATION PHASE
        if self.mode == "CALIBRATING":
            return self._run_calibration(grid, agent_rc)

        # 3. PERSPECTIVE BIDDING (Sanitized: No side effects yet)
        bids = []

        # Newton (Survival/Nav)
        if self.mode == "AVATAR":
            # Returns (Score, Action, Target, Owner)
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols))

        # Euclid (Geometry/Completion)
        bids.append(self._perspective_euclid(grid, rows, cols))

        # Skinner (Causality/Probing)
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # Filter None
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids:
            return self._perspective_chaos()

        # 4. RECURSIVE CRITIQUE (The Cortex)
        # Newton validates targets for reachability
        vetted_bids = []
        for score, action, target, owner in valid_bids:
            final_score = score

            # Critique: "Can I reach this click target?"
            if target and agent_rc and owner in ["Euclid", "Skinner"]:
                path = self._astar_soft(grid, agent_rc, target, rows, cols)
                if not path:
                    # VETO: Target is behind a wall or unreachable
                    final_score *= 0.1

            # Critique: "Am I stuck?" -> Boost Skinner
            if owner == "Skinner" and self.stuck_counter > 2:
                final_score *= 2.0

            vetted_bids.append((final_score, action, target, owner))

        # 5. SELECTION & SIDE EFFECTS
        vetted_bids.sort(key=lambda x: x[0], reverse=True)
        best_bid = vetted_bids[0]
        score, action, target, owner = best_bid

        # Conflict Resolution: If tie between Pattern & Nav, prioritize Nav (Survival)
        if len(vetted_bids) > 1:
            runner = vetted_bids[1]
            if (score - runner[0] < 0.05) and (owner == "Euclid") and (runner[3] == "Newton"):
                best_bid = runner
                score, action, target, owner = best_bid

        # Apply Side Effects (Memory Update) ONLY for the winner
        if target:
            if action == self.ACT_CLICK:
                self.interactions.add(target)
                self._set_click_payload(target) # Sets self.action_data
            elif action == self.ACT_USE:
                self.interactions.add(target)

        self.last_action = action
        return action

    # --- PERSPECTIVES ---

    def _perspective_newton(self, grid, agent_rc, rows, cols):
        """Navigation: Soft A* with Dynamic Obstacles"""
        if not agent_rc: return (0.0, None, None, "Newton")

        targets = self._scan_targets(grid, agent_rc)
        if not targets:
            # Frontier Logic
            frontier = self._find_frontier(grid, agent_rc, rows, cols)
            if frontier: return (0.6, frontier[0], None, "Newton")
            return (0.0, None, None, "Newton")

        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue
            path = self._astar_soft(grid, agent_rc, t_rc, rows, cols)
            if path: return (0.95, path[0], None, "Newton")
            else: self.bad_goals.add(t_rc)

        return (0.0, None, None, "Newton")

    def _perspective_euclid(self, grid, rows, cols):
        """Geometry: Vector Extension + Symmetry"""
        # 1. Vector Extension
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            return (0.9, self.ACT_CLICK, ext, "Euclid")

        # 2. Symmetry Completion (v4.0 Feature)
        sym_target = self._detect_symmetry_completion(grid, rows, cols)
        if sym_target:
             return (0.85, self.ACT_CLICK, sym_target, "Euclid")

        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols):
        """Causality: Weighted Interaction"""
        # 1. Local Interaction (Spacebar)
        if agent_rc:
            adj = self._scan_adjacent(grid, agent_rc, rows, cols)
            for r, c in adj:
                val = grid[r, c]
                if val != 0 and (r,c) not in self.interactions:
                    # Boost score if this object type has reacted before
                    score = 0.7 + (self.causal_history[val] * 0.2)
                    return (min(0.99, score), self.ACT_USE, (r,c), "Skinner")

        # 2. Global Clicking
        unique, counts = np.unique(grid, return_counts=True)
        indices = np.argsort(counts)
        sorted_vals = unique[indices]

        for val in sorted_vals:
            if val == 0: continue
            matches = np.argwhere(grid == val)
            for r, c in matches:
                if (r, c) not in self.interactions:
                    # Weighted by Causal History
                    score = 0.5 + (self.causal_history[val] * 0.4)
                    return (min(0.99, score), self.ACT_CLICK, (r,c), "Skinner")

        return (0.0, None, None, "Skinner")

    def _perspective_chaos(self):
        return random.choice([self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT])

    # --- SUBSTRATE LOGIC ---

    def _update_physics(self, grid, agent_rc, current_hash):
        # 1. Identity Check
        if self.mode == "AVATAR" and self.last_grid is not None and self.last_pos is not None:
             if current_hash != self.last_state[1]:
                 # Causal Learning: Record what changed
                 diff = grid != self.last_grid
                 changed_vals = grid[diff]
                 for v in changed_vals:
                     if v != 0: self.causal_history[v] += 1.0 # Boost influence

                 # Identity Correction
                 if agent_rc is None or agent_rc == self.last_pos:
                      new_c = self._detect_moving_color(self.last_grid, grid, self.last_pos)
                      if new_c is not None: self.agent_color = new_c

        # 2. Wall/Goal Reset
        if abs(current_hash - self.inventory_hash) > 0:
            self.walls.clear()
            self.bad_goals.clear()
        self.inventory_hash = current_hash

        # 3. Collision
        if self.mode == "AVATAR" and self.last_pos and self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
             if agent_rc == self.last_pos and current_hash == self.last_state[1]:
                 self.stuck_counter += 1
                 if self.stuck_counter > 0:
                     tr, tc = self._get_target(self.last_pos, self.last_action)
                     self.walls.add((tr, tc))
             else:
                 self.stuck_counter = 0

        self.last_grid = grid.copy()
        self.last_state = (agent_rc, current_hash)
        self.last_pos = agent_rc

    def _run_calibration(self, grid, agent_rc):
        # If we moved or grid changed (physics response)
        if self.last_pos and agent_rc and agent_rc != self.last_pos:
            self.mode = "AVATAR"
            return self._omega_executive(FrameData(grid.tolist(), None, GameState.PLAYING))

        # Check grid change (for ls20 fix)
        if self.last_grid is not None:
             if not np.array_equal(grid, self.last_grid):
                 self.mode = "AVATAR" # World changed -> Physics
                 return self._omega_executive(FrameData(grid.tolist(), None, GameState.PLAYING))

        if self.calibration_queue:
            return self.calibration_queue.popleft()

        self.mode = "SCIENTIST"
        return self._omega_executive(FrameData(grid.tolist(), None, GameState.PLAYING))

    # --- ALGORITHMIC HELPERS ---

    def _detect_symmetry_completion(self, grid, rows, cols):
        """Detects missing pixels in Mirror Symmetry (Horizontal)."""
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mirror_c = cols - 1 - c
                val_l = grid[r, c]
                val_r = grid[r, mirror_c]

                # If Left exists and Right is empty -> Click Right
                if val_l != 0 and val_r == 0:
                    if (r, mirror_c) not in self.interactions:
                        return (r, mirror_c)
                # If Right exists and Left is empty -> Click Left
                if val_r != 0 and val_l == 0:
                    if (r, c) not in self.interactions:
                        return (r, c)
        return None

    def _find_vector_extension(self, grid, rows, cols):
        non_zeros = np.argwhere(grid != 0)
        if len(non_zeros) > 80: return None
        by_color = defaultdict(list)
        for r, c in non_zeros: by_color[grid[r,c]].append((r,c))

        for color, coords in by_color.items():
            if len(coords) < 2: continue
            coords.sort()
            for i in range(len(coords)-1):
                r1, c1 = coords[i]
                r2, c2 = coords[i+1]
                dr, dc = r2-r1, c2-c1
                if abs(dr) > 2 or abs(dc) > 2: continue

                # Project forward
                nr, nc = r2+dr, c2+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 0 and (nr, nc) not in self.interactions:
                        return (nr, nc)
        return None

    def _astar_soft(self, grid, start, end, rows, cols):
        pq = [(0, 0, start, [])]
        best_g = {start: 0}
        while pq:
            f, steps, curr, path = heapq.heappop(pq)
            if curr == end: return path
            if steps > 300: break

            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in self.walls: continue
                    val = grid[nr, nc]
                    cost = 1 if val == 0 or (nr, nc) == end else 5
                    new_g = steps + cost
                    if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                        best_g[(nr, nc)] = new_g
                        h = abs(nr - end[0]) + abs(nc - end[1])
                        heapq.heappush(pq, (new_g + h, new_g, (nr, nc), path + [act]))
        return None

    def _locate_agent(self, latest, grid):
        if hasattr(latest, 'agent_pos') and latest.agent_pos and latest.agent_pos != (-1, -1):
            return (latest.agent_pos[1], latest.agent_pos[0])
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0: return tuple(coords[0])
        unique, counts = np.unique(grid, return_counts=True)
        indices = np.argsort(counts)
        for i in indices:
            c, count = unique[i], counts[i]
            if count < 50:
                coords = np.argwhere(grid == c)
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
        diff = prev_grid != curr_grid
        if not np.any(diff): return None
        r, c = last_pos
        neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
        rows, cols = curr_grid.shape
        candidates = []
        for nr, nc in neighbors:
             if 0 <= nr < rows and 0 <= nc < cols:
                 val = curr_grid[nr, nc]
                 if val != 0 and val != prev_grid[nr, nc]: candidates.append(val)
        if candidates: return Counter(candidates).most_common(1)[0][0]
        return None

    def _find_frontier(self, grid, start, rows, cols):
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

    def _scan_adjacent(self, grid, rc, rows, cols):
        r, c = rc
        adj = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols: adj.append((nr, nc))
        return adj

    def _set_click_payload(self, rc):
        r, c = rc
        payload = {'x': int(c), 'y': int(r)}
        self.action_data = payload
        try: self.ACT_CLICK.set_data(payload)
        except: pass

    def _get_target(self, rc, action):
        r, c = rc
        if action == self.ACT_UP: return r-1, c
        if action == self.ACT_DOWN: return r+1, c
        if action == self.ACT_LEFT: return r, c-1
        if action == self.ACT_RIGHT: return r, c+1
        return r, c
