import numpy as np
import random
import heapq
from collections import deque, Counter, defaultdict
from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- STIGMERGIC MEMORY ---
class GlobalNexus:
    CONTROLS = {}
    PERSPECTIVE_WEIGHTS = defaultdict(lambda: 1.0)
    USE_SCORES = defaultdict(lambda: 1.0)
    CLICK_SCORES = defaultdict(lambda: 1.0)

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v9.1-COSMOS (KEVIN_KULL)
    The 'Phase 1' Standard Agent.

    CAPABILITIES:
    1. COSMOS LOOP BREAKER: Detects 'State Stasis' (Position + Grid Hash).
    2. RECOVERY INJECTION: Forces entropy (Chaos) when stuck > 5.
    3. DELTA IDENTITY: Locates agent via grid change if color tracking fails.
    4. SPLIT CAUSALITY: Separate weights for USE vs CLICK.
    """

    MAX_ACTIONS = 400 # CRITICAL: Extended runway for recovery

    RAW_ACTIONS = [
        GameAction.ACTION1, GameAction.ACTION2,
        GameAction.ACTION3, GameAction.ACTION4
    ]
    DEFAULT_CONTROLS = {
        GameAction.ACTION1: (-1, 0), GameAction.ACTION2: (1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)
    }

    ACT_USE = getattr(GameAction, 'ACTION5', GameAction.ACTION5)
    ACT_CLICK = getattr(GameAction, 'ACTION6', GameAction.ACTION6)
    ACT_INTERACT = getattr(GameAction, 'ACTION7', GameAction.ACTION5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_map = {}
        self.walls = set()
        self.bad_goals = set()
        self.interactions = set()

        self.mode = "HANDSHAKE"
        self.handshake_queue = deque(self.RAW_ACTIONS)
        self.last_pos = None
        self.last_grid = None
        self.last_action = None
        self.last_target_val = None
        self.agent_color = None
        self.stuck_counter = 0
        self.lost_counter = 0
        self.action_data = {}

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        self.action_data = {}
        if not self.control_map and GlobalNexus.CONTROLS:
            self.control_map = GlobalNexus.CONTROLS.copy()
            self.mode = "AVATAR"

        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET
        try:
            return self._cosmos_loop(latest)
        except Exception:
            return random.choice(self.RAW_ACTIONS)

    def _cosmos_loop(self, latest: FrameData) -> GameAction:
        # 1. PERCEPT
        grid = self._parse_grid(latest)
        rows, cols = grid.shape
        current_hash = np.sum(grid)

        # 2. DELTA IDENTITY (Safety Net)
        agent_rc = self._locate_agent(latest, grid)
        if agent_rc is None and self.last_grid is not None:
             # If grid size matches, check for delta
             if grid.shape == self.last_grid.shape:
                 delta_rc = self._find_center_of_change(self.last_grid, grid)
                 if delta_rc:
                     agent_rc = delta_rc
                     self.agent_color = grid[agent_rc]

        if agent_rc is None: self.lost_counter += 1
        else: self.lost_counter = 0

        # 3. PHYSICS UPDATE (The Fix: Progress-Decoupled)
        self._update_physics(grid, agent_rc, current_hash)

        # 4. L0: HANDSHAKE
        if self.mode == "HANDSHAKE":
            return self._run_handshake(grid, agent_rc)

        # 5. L1: PARLIAMENT
        bids = []

        # [Blind Pilot] (Priority 0.99 if lost)
        if self.mode == "AVATAR" and agent_rc is None:
            bid = 0.99 if self.lost_counter < 40 else 0.0
            bids.append((bid, self._perspective_chaos(), None, "BlindPilot"))

        # [Newton] Nav
        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols))
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        # [Euclid] Pattern
        bids.append(self._perspective_euclid(grid, rows, cols))

        # [Skinner] Interact
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # [Recovery] (Triggers on Global Stasis)
        bids.append(self._perspective_recovery())

        # 6. L2: EXECUTIVE
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids: return self._perspective_chaos()

        weighted_bids = []
        for score, action, target, owner in valid_bids:
            w_score = score * GlobalNexus.PERSPECTIVE_WEIGHTS[owner]
            # Safety: Don't walk into known walls (unless Recovery overrides)
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            weighted_bids.append((w_score, action, target, owner))

        weighted_bids.sort(key=lambda x: x[0], reverse=True)
        best_bid = weighted_bids[0]
        score, action, target, owner = best_bid

        # 7. EXECUTION
        if target:
            val = 1
            if len(target) > 2: val = target[2]
            else:
                try: val = grid[target[0], target[1]]
                except: pass

            self._set_payload(target[:2], val)
            self.last_target_val = val
            if action in [self.ACT_CLICK, self.ACT_USE]:
                self.interactions.add(target[:2] if len(target)>2 else target)
        else:
            self.last_target_val = None

        self.last_action = action
        return action

    # --- PERSPECTIVES ---

    def _perspective_newton(self, grid, agent_rc, rows, cols):
        if not agent_rc: return (0.0, None, None, "Newton")
        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue
            # Allow push (cost=5)
            path = self._astar(grid, agent_rc, t_rc, rows, cols, allow_push=True)
            if path: return (0.95, path[0], t_rc, "Newton")
            else: self.bad_goals.add(t_rc)

        frontier = self._find_frontier(grid, agent_rc, rows, cols)
        if frontier: return (0.6, frontier[0], None, "Newton")
        return (0.0, None, None, "Newton")

    def _perspective_euclid(self, grid, rows, cols):
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            target, color = ext
            return (0.92, self.ACT_CLICK, (target[0], target[1], color), "Euclid")
        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols):
        if agent_rc:
            adj = self._scan_adjacent(grid, agent_rc, rows, cols)
            for r, c in adj:
                val = grid[r, c]
                if val != 0 and (r,c) not in self.interactions:
                    score = 0.8 * GlobalNexus.USE_SCORES[val]
                    return (score, self.ACT_USE, (r,c), "Skinner")

        # Click Gating: Allow if lost or in Scientist Mode
        if self.mode != "AVATAR" or self.lost_counter > 20:
            unique, counts = np.unique(grid, return_counts=True)
            for val in unique[np.argsort(counts)]:
                if val == 0: continue
                matches = np.argwhere(grid == val)
                for r, c in matches:
                    if (r, c) not in self.interactions:
                         score = 0.6 * GlobalNexus.CLICK_SCORES[val]
                         return (score, self.ACT_CLICK, (r,c, val), "Skinner")
        return (0.0, None, None, "Skinner")

    def _perspective_fusion(self, grid, agent_rc, rows, cols):
        if not agent_rc: return (0.0, None, None, "Fusion")
        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            if dist > 1 and t_rc not in self.interactions:
                val = grid[t_rc]
                adj = self._scan_adjacent(grid, t_rc, rows, cols)
                for near_rc in adj:
                    if grid[near_rc] == 0:
                        path = self._astar(grid, agent_rc, near_rc, rows, cols)
                        if path:
                            relevance = max(GlobalNexus.USE_SCORES[val], GlobalNexus.CLICK_SCORES[val])
                            score = 0.94 * relevance
                            return (score, path[0], t_rc, "Fusion")
        return (0.0, None, None, "Fusion")

    def _perspective_recovery(self):
        # COSMOS FIX: Triggers on Global Stasis (stuck_counter > 2)
        if self.stuck_counter > 2:
            if self.stuck_counter > 5:
                # CHAOS INJECTION (Priority 3.0)
                return (3.0, self._perspective_chaos(), None, "Recovery")
            # Try interaction
            action = self.ACT_USE if self.stuck_counter % 2 != 0 else self.ACT_INTERACT
            return (0.98, action, None, "Recovery")
        return (0.0, None, None, "Recovery")

    def _perspective_chaos(self):
        if self.control_map: return random.choice(list(self.control_map.keys()))
        return random.choice(self.RAW_ACTIONS)

    # --- SUBSTRATE ---

    def _run_handshake(self, grid, agent_rc):
        if self.last_pos and agent_rc and self.last_action in self.RAW_ACTIONS:
            dr, dc = agent_rc[0] - self.last_pos[0], agent_rc[1] - self.last_pos[1]
            if dr != 0 or dc != 0:
                self.control_map[self.last_action] = (dr, dc)
                GlobalNexus.CONTROLS[self.last_action] = (dr, dc)
        self.last_pos = agent_rc

        if self.handshake_queue:
            return self.handshake_queue.popleft()
        else:
            self.mode = "AVATAR"
            # PARTIAL MERGE: Ensures 4-way control even on partial calibration
            if not self.control_map:
                self.control_map = self.DEFAULT_CONTROLS.copy()
            else:
                for k, v in self.DEFAULT_CONTROLS.items():
                    if k not in self.control_map: self.control_map[k] = v
            GlobalNexus.CONTROLS = self.control_map.copy()
            return random.choice(self.RAW_ACTIONS)

    def _update_physics(self, grid, agent_rc, current_hash):
        # 1. FEEDBACK DECAY
        if self.last_action in [self.ACT_CLICK, self.ACT_USE] and self.last_grid is not None:
             success = (current_hash != np.sum(self.last_grid))
             val = self.last_target_val
             if val is not None:
                 if success:
                     if self.last_action == self.ACT_USE: GlobalNexus.USE_SCORES[val] *= 1.2
                     if self.last_action == self.ACT_CLICK: GlobalNexus.CLICK_SCORES[val] *= 1.2
                 else:
                     if self.last_action == self.ACT_USE: GlobalNexus.USE_SCORES[val] *= 0.8
                     if self.last_action == self.ACT_CLICK: GlobalNexus.CLICK_SCORES[val] *= 0.8

        # 2. PROGRESS-DECOUPLED STUCK DETECTION (The Fix)
        state_changed = (current_hash != np.sum(self.last_grid)) if self.last_grid is not None else True
        moved = (agent_rc != self.last_pos) if self.last_pos is not None and agent_rc is not None else False

        if not state_changed and not moved:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # Wall Learning
        if self.mode == "AVATAR" and self.last_pos and agent_rc == self.last_pos:
             if self.last_action in self.control_map:
                 dr, dc = self.control_map[self.last_action]
                 tr, tc = self.last_pos[0]+dr, self.last_pos[1]+dc
                 self.walls.add((tr, tc))
                 if self.stuck_counter > 8: self.agent_color = None

        self.last_pos = agent_rc
        self.last_grid = grid.copy()

    # --- HELPERS ---

    def _find_center_of_change(self, prev, curr):
        if prev.shape != curr.shape: return None
        diff = prev != curr
        coords = np.argwhere(diff)
        if len(coords) == 0: return None
        avg_r = int(np.mean(coords[:, 0]))
        avg_c = int(np.mean(coords[:, 1]))
        return (avg_r, avg_c)

    def _parse_grid(self, latest):
        raw = np.array(latest.frame)
        if raw.ndim == 2: return raw
        if raw.ndim == 3: return raw[-1]
        return raw.reshape((int(np.sqrt(raw.size)), -1))

    def _astar(self, grid, start, end, rows, cols, allow_push=False):
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
                    val = grid[nr, nc]
                    is_walk = (val == 0) or ((nr, nc) == end)
                    if allow_push and (nr, nc) not in self.walls: is_walk = True
                    if (nr, nc) in self.walls: is_walk = False
                    if is_walk:
                        cost = 1 if val == 0 else 5
                        ng = steps + cost
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
        indices = np.argsort(counts)
        for i in indices:
            c, count = unique[i], counts[i]
            if c == 0 and count > (grid.size // 2): continue
            coords = np.argwhere(grid == c)
            if len(coords) > 0: return tuple(coords[0])
        return None

    def _detect_moving_color(self, prev, curr):
        if prev.shape != curr.shape: return None
        diff = prev != curr
        if not np.any(diff): return None
        cands = []
        diff_coords = np.argwhere(diff)
        for r, c in diff_coords:
            val = curr[r, c]
            prev_val = prev[r, c]
            if val != 0 and prev_val == 0: cands.append(val)
        if cands: return Counter(cands).most_common(1)[0][0]
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
        try: self.ACT_CLICK.action_data = payload
        except: pass

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
