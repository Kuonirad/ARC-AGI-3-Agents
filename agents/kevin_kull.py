import numpy as np
import random
import heapq
from collections import deque, Counter, defaultdict
from collections import deque, Counter
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
    NOVA-M-COP v3.0-KEVIN_KULL
    The 'Optimistic' Polymath.

    PHILOSOPHY:
    - Optimistic Physics: Assume everything is walkable until we crash.
    - Wiggle-Calibration: Try all 4 directions before giving up on having a body.
    - Dynamic Identity: Tracking logic that handles color/shape shifts (Keys, Portals).
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
        self.action_data = {} # Reset
        # Reset payload every tick to prevent leakage
        self.action_data = {}

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

        # 3. Frontier (Explore unknown)
        # 3. Frontier Exploration (If no goals)
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
                    return self._execute_click((r, c))

        # --- PHASE 3: ENTROPY (Random) ---
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
        payload = {'x': int(c), 'y': int(r)}
        self.action_data = payload

        # SHIM: Attach data to the Enum Member
        payload = {'x': int(c), 'y': int(r)}

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
        diff = prev_grid != curr_grid
        if not np.any(diff): return None
        r, c = last_pos
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
                 if val != 0 and val != prev_grid[nr, nc]: candidates.append(val)
                 if val != 0 and val != prev_grid[nr, nc]:
                     candidates.append(val)
        if candidates: return Counter(candidates).most_common(1)[0][0]
        return None

    def _find_frontier(self, grid, start, rows, cols):
        """BFS to unknown"""

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
