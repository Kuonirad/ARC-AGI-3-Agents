import numpy as np
import random
import heapq
from collections import deque, Counter, defaultdict
from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- STIGMERGIC MEMORY (Fleet-Wide) ---
class GlobalNexus:
    """
    Shared knowledge base for all KevinKull instances.
    - Shares PHYSICS (Controls) -> Persistent across levels.
    - Shares CAUSALITY (Weights) -> Persistent across levels.
    - DOES NOT Share MAPS (Walls) -> Reset per level (Safety).
    """
    CONTROLS = {}
    PERSPECTIVE_WEIGHTS = defaultdict(lambda: 1.0)
    CAUSAL_SCORES = defaultdict(lambda: 1.0) # Object ID -> Interaction Value

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v6.0-NEXUS (KEVIN_KULL)
    The 'Autotelic Alignment' Agent.

    INTEGRATION TIER:
    1. PRODIGY HANDSHAKE: Dynamically learns Input->Vector mapping.
    2. NUCLID FUSION: Synthesizes 'Move-to-Interact' plans.
    3. SAFE STIGMERGY: Fleet-wide physics sync.
    4. RECURSIVE CRITIQUE: Executive veto for unreachable targets.
    """

    # Base Actions
    RAW_ACTIONS = [
        GameAction.ACTION1, GameAction.ACTION2,
        GameAction.ACTION3, GameAction.ACTION4
    ]
    ACT_USE = getattr(GameAction, 'ACTION5', GameAction.ACTION5)
    ACT_CLICK = getattr(GameAction, 'ACTION6', GameAction.ACTION6)
    ACT_INTERACT = getattr(GameAction, 'ACTION7', GameAction.ACTION5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # --- Local Memory (Per Level) ---
        self.control_map = {}     # Synced from Nexus
        self.walls = set()
        self.bad_goals = set()
        self.interactions = set()

        # --- State ---
        self.mode = "HANDSHAKE"
        self.handshake_queue = deque(self.RAW_ACTIONS)
        self.last_pos = None
        self.last_grid = None
        self.last_action = None
        self.last_perspective = None
        self.agent_color = None
        self.stuck_counter = 0
        self.inventory_hash = 0
        self.action_data = {}

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        self.action_data = {}
        # Sync with Fleet
        if not self.control_map and GlobalNexus.CONTROLS:
            self.control_map = GlobalNexus.CONTROLS.copy()
            self.mode = "AVATAR"

        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET
        try:
            return self._nexus_loop(latest)
        except Exception:
            return random.choice(self.RAW_ACTIONS)

    def _nexus_loop(self, latest: FrameData) -> GameAction:
        # 1. PERCEPT
        grid = self._parse_grid(latest)
        rows, cols = grid.shape
        agent_rc = self._locate_agent(latest, grid)
        current_hash = np.sum(grid)

        # 2. PHYSICS UPDATE & META-LEARNING
        self._update_physics(grid, agent_rc, current_hash)

        # 3. L0: HANDSHAKE (Calibration)
        if self.mode == "HANDSHAKE":
            return self._run_handshake(grid, agent_rc)

        # 4. L1: PARLIAMENT OF PERSPECTIVES (Bidding)
        bids = []

        # [Newton] Navigation
        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols))

        # [Euclid] Pattern
        bids.append(self._perspective_euclid(grid, rows, cols))

        # [Skinner] Interaction
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # [Fusion] 'Nuclid' (Nav-to-Interact)
        if self.mode == "AVATAR":
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        # 5. L2: NEXUS EXECUTIVE (Critique & Select)
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids:
            return self._perspective_chaos()

        weighted_bids = []
        for score, action, target, owner in valid_bids:
            # Meta-Weight Application
            w_score = score * GlobalNexus.PERSPECTIVE_WEIGHTS[owner]

            # Recursive Critique: Safety
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0 # Veto known walls

            weighted_bids.append((w_score, action, target, owner))

        weighted_bids.sort(key=lambda x: x[0], reverse=True)
        best_bid = weighted_bids[0]
        score, action, target, owner = best_bid

        # 6. EXECUTION & SIDE EFFECTS
        if target:
            if action == self.ACT_CLICK:
                self.interactions.add(target[:2] if len(target)>2 else target)
                val = 1
                if len(target) > 2: val = target[2]
                self._set_payload(target[:2] if len(target)>2 else target, val)
            elif action == self.ACT_USE:
                self.interactions.add(target)

        # Vitruvian Stuck Recovery Override
        if self.stuck_counter > 2:
            action = self.ACT_USE if self.stuck_counter == 3 else self.ACT_INTERACT
            owner = "Recovery"

        self.last_action = action
        self.last_perspective = owner
        return action

    # --- PERSPECTIVES ---

    def _perspective_newton(self, grid, agent_rc, rows, cols):
        """Navigation: Soft A* to Rare Objects."""
        if not agent_rc: return (0.0, None, None, "Newton")

        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue
            path = self._astar(grid, agent_rc, t_rc, rows, cols)
            if path: return (0.95, path[0], t_rc, "Newton")
            else: self.bad_goals.add(t_rc)

        frontier = self._find_frontier(grid, agent_rc, rows, cols)
        if frontier: return (0.6, frontier[0], None, "Newton")
        return (0.0, None, None, "Newton")

    def _perspective_euclid(self, grid, rows, cols):
        """Pattern: Vector Extension."""
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            target, color = ext
            return (0.92, self.ACT_CLICK, (target[0], target[1], color), "Euclid")
        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols):
        """Interaction: Probing."""
        # Local Use
        if agent_rc:
            adj = self._scan_adjacent(grid, agent_rc, rows, cols)
            for r, c in adj:
                val = grid[r, c]
                if val != 0 and (r,c) not in self.interactions:
                    score = 0.8 * GlobalNexus.CAUSAL_SCORES[val]
                    return (score, self.ACT_USE, (r,c), "Skinner")

        # Global Click
        unique, counts = np.unique(grid, return_counts=True)
        for val in unique[np.argsort(counts)]:
            if val == 0: continue
            matches = np.argwhere(grid == val)
            for r, c in matches:
                if (r, c) not in self.interactions:
                     score = 0.6 * GlobalNexus.CAUSAL_SCORES[val]
                     return (score, self.ACT_CLICK, (r,c, val), "Skinner")
        return (0.0, None, None, "Skinner")

    def _perspective_fusion(self, grid, agent_rc, rows, cols):
        """Fusion: Move to where Skinner wants to interact."""
        if not agent_rc: return (0.0, None, None, "Fusion")

        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            # If target is interesting but distant
            if dist > 1 and t_rc not in self.interactions:
                val = grid[t_rc]
                # Check adjacent spots
                adj = self._scan_adjacent(grid, t_rc, rows, cols)
                for near_rc in adj:
                    if grid[near_rc] == 0: # Open spot near target
                        path = self._astar(grid, agent_rc, near_rc, rows, cols)
                        if path:
                            # High confidence: Navigating to enable Interaction
                            score = 0.94 * GlobalNexus.CAUSAL_SCORES[val]
                            return (score, path[0], t_rc, "Fusion")
        return (0.0, None, None, "Fusion")

    def _perspective_chaos(self):
        if self.control_map: return random.choice(list(self.control_map.keys()))
        return random.choice(self.RAW_ACTIONS)

    # --- SUBSTRATE ---

    def _run_handshake(self, grid, agent_rc):
        # 1. Physics Check
        if self.last_pos and self.last_action in self.RAW_ACTIONS:
            # Check for Identity Loss (Grid changed but agent didn't move)
            current_hash = np.sum(grid)
            prev_hash = np.sum(self.last_grid) if self.last_grid is not None else 0

            if current_hash != prev_hash and (agent_rc is None or agent_rc == self.last_pos):
                 if self.last_grid is not None:
                     new_c = self._detect_moving_color(self.last_grid, grid, self.last_pos)
                     if new_c:
                         self.agent_color = new_c
                         agent_rc = self._locate_agent(FrameData([grid.tolist()], None, GameState.PLAYING), grid)

            # Check for Motion
            if agent_rc:
                dr, dc = agent_rc[0] - self.last_pos[0], agent_rc[1] - self.last_pos[1]
                if dr != 0 or dc != 0:
                    self.control_map[self.last_action] = (dr, dc)
                    GlobalNexus.CONTROLS[self.last_action] = (dr, dc)

        self.last_pos = agent_rc
        self.last_grid = grid.copy()

        if self.handshake_queue:
            action = self.handshake_queue.popleft()
            self.last_action = action
            return action
        else:
            self.mode = "AVATAR" if self.control_map else "SCIENTIST"
            return random.choice(self.RAW_ACTIONS)

    def _update_physics(self, grid, agent_rc, current_hash):
        # Meta-Learning: Reward success
        if self.mode == "AVATAR" and self.last_grid is not None:
             if current_hash != np.sum(self.last_grid): # World Changed
                 GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] *= 1.1
                 # Boost causal score of objects we interacted with
                 if self.last_action in [self.ACT_CLICK, self.ACT_USE] and self.interactions:
                     pass # Hard to track exact object, but we reward the attempt

                 # Identity Check
                 if agent_rc is None or agent_rc == self.last_pos:
                     new_c = self._detect_moving_color(self.last_grid, grid, self.last_pos)
                     if new_c: self.agent_color = new_c

        # Wall Detection
        if self.mode == "AVATAR" and self.last_pos and agent_rc == self.last_pos:
             if self.last_action in self.control_map:
                 self.stuck_counter += 1
                 if self.stuck_counter > 1:
                     dr, dc = self.control_map[self.last_action]
                     tr, tc = self.last_pos[0]+dr, self.last_pos[1]+dc
                     self.walls.add((tr, tc))
        else:
             self.stuck_counter = 0

        self.last_pos = agent_rc
        self.last_grid = grid.copy()

    # --- HELPERS ---

    def _parse_grid(self, latest):
        raw = np.array(latest.frame)
        if raw.ndim == 2: return raw
        if raw.ndim == 3: return raw[-1]
        return raw.reshape((int(np.sqrt(raw.size)), -1))

    def _astar(self, grid, start, end, rows, cols):
        if not self.control_map: return None
        pq = [(0, 0, start, [])]
        best_g = {start: 0}
        while pq:
            f, steps, curr, path = heapq.heappop(pq)
            if curr == end: return path
            if steps > 200: break
            r, c = curr
            for act, (dr, dc) in self.control_map.items():
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    is_walk = (grid[nr, nc] == 0) or ((nr, nc) == end)
                    if (nr, nc) in self.walls: is_walk = False
                    if is_walk:
                        ng = steps + 1
                        if (nr, nc) not in best_g or ng < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = ng
                            h = abs(nr-end[0]) + abs(nc-end[1])
                            heapq.heappush(pq, (ng+h, ng, (nr, nc), path+[act]))
        return None

    def _locate_agent(self, latest, grid):
        if hasattr(latest, 'agent_pos') and latest.agent_pos and latest.agent_pos != (-1, -1):
            return (latest.agent_pos[1], latest.agent_pos[0])
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0: return tuple(coords[0])
        unique, counts = np.unique(grid, return_counts=True)
        # Sort by rarity, check if non-background
        indices = np.argsort(counts)
        for i in indices:
            c, count = unique[i], counts[i]
            # Heuristic: Agent is rare. Background (0) is common.
            # If 0 is background, count will be high. Skip it.
            if c == 0 and count > (grid.size // 2): continue
            coords = np.argwhere(grid == c)
            if len(coords) > 0: return tuple(coords[0])
        return None

    def _find_vector_extension(self, grid, rows, cols):
        non_zeros = np.argwhere(grid != 0)
        if len(non_zeros) > 100: return None
        by_color = defaultdict(list)
        for r, c in non_zeros: by_color[grid[r,c]].append((r,c))
        for color, coords in by_color.items():
            if len(coords) < 2: continue
            coords.sort()
            for i in range(len(coords)-1):
                r1, c1 = coords[i]; r2, c2 = coords[i+1]
                dr, dc = r2-r1, c2-c1
                if abs(dr) > 3 or abs(dc) > 3: continue
                nr, nc = r2+dr, c2+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 0 and (nr, nc) not in self.interactions:
                        return ((nr, nc), color)
        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 50) & (unique != 0)]
        targets = []
        matches = np.argwhere(np.isin(grid, rare))
        for r, c in matches:
            if (r, c) != start_rc:
                targets.append((abs(r-start_rc[0]) + abs(c-start_rc[1]), (r, c)))
        targets.sort()
        return targets

    def _find_frontier(self, grid, start, rows, cols):
        if not self.control_map: return None
        q = deque([(start, [])])
        seen = {start}
        steps = 0
        while q:
            steps += 1;
            if steps > 200: break
            curr, path = q.popleft(); r, c = curr
            for act, (dr, dc) in self.control_map.items():
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

    def _set_payload(self, rc, val):
        r, c = rc
        payload = {'x': int(c), 'y': int(r), 'value': int(val)}
        self.action_data = payload
        try: self.ACT_CLICK.set_data(payload)
        except: pass

    def _detect_moving_color(self, prev, curr, last_pos):
        if prev.shape != curr.shape: return None
        diff = prev != curr
        if not np.any(diff): return None

        # Scan entire diff for new colors appearing
        # The agent moved TO a location, so that location changed value.
        changed_vals = curr[diff]
        cands = [v for v in changed_vals if v != 0]

        if cands: return Counter(cands).most_common(1)[0][0]
        return None
