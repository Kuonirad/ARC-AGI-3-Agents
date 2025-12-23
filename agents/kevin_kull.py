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
    CAUSAL_SCORES = defaultdict(lambda: 1.0)

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v7.0-ZENITH (KEVIN_KULL)
    The 'Motion-Derived Identity' Agent.

    CRITICAL FIXES:
    1. MOTION IDENTITY: If grid changes during handshake, we deduce Agent Color from the delta.
    2. SCIENTIST GATING: In AVATAR mode, Global Clicks are BANNED. Only Local Interaction allowed.
    3. DEFAULT CONTROLS: If Handshake fails, we guess standard arrow keys instead of giving up.
    """

    RAW_ACTIONS = [
        GameAction.ACTION1, GameAction.ACTION2,
        GameAction.ACTION3, GameAction.ACTION4
    ]
    # Default Assumptions (Hail Mary)
    DEFAULT_CONTROLS = {
        GameAction.ACTION1: (-1, 0), # Up
        GameAction.ACTION2: (1, 0),  # Down
        GameAction.ACTION3: (0, -1), # Left
        GameAction.ACTION4: (0, 1)   # Right
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

        # State
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
        # Fleet Sync
        if not self.control_map and GlobalNexus.CONTROLS:
            self.control_map = GlobalNexus.CONTROLS.copy()
            self.mode = "AVATAR"

        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET
        try:
            return self._zenith_loop(latest)
        except Exception:
            return random.choice(self.RAW_ACTIONS)

    def _zenith_loop(self, latest: FrameData) -> GameAction:
        # 1. PERCEPT
        grid = self._parse_grid(latest)
        rows, cols = grid.shape
        current_hash = np.sum(grid)

        # 2. MOTION-DERIVED IDENTITY (The Fix)
        # If we don't know who we are, check if we moved
        if self.agent_color is None and self.last_grid is not None:
             if current_hash != np.sum(self.last_grid):
                 # World changed. Find the delta.
                 new_c = self._detect_moving_color(self.last_grid, grid)
                 if new_c is not None:
                     self.agent_color = new_c # Found ourselves!

        agent_rc = self._locate_agent(latest, grid)

        # 3. PHYSICS UPDATE
        self._update_physics(grid, agent_rc, current_hash)

        # 4. L0: HANDSHAKE
        if self.mode == "HANDSHAKE":
            return self._run_handshake(grid, agent_rc)

        # 5. L1: PARLIAMENT
        bids = []

        # Newton (Nav)
        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols))
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        # Euclid (Pattern) - Always active for vectors
        bids.append(self._perspective_euclid(grid, rows, cols))

        # Skinner (Interact) - GATED
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # 6. L2: EXECUTIVE
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids:
            return self._perspective_chaos()

        weighted_bids = []
        for score, action, target, owner in valid_bids:
            w_score = score * GlobalNexus.PERSPECTIVE_WEIGHTS[owner]
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            weighted_bids.append((w_score, action, target, owner))

        weighted_bids.sort(key=lambda x: x[0], reverse=True)
        best_bid = weighted_bids[0]
        score, action, target, owner = best_bid

        # 7. EXECUTION
        if target:
            # Extract coords
            t_rc = target[:2] if len(target) > 2 else target

            if action == self.ACT_CLICK:
                self.interactions.add((t_rc, action))
                val = 1
                if len(target) > 2: val = target[2]
                self._set_payload(t_rc, val)
            elif action == self.ACT_USE:
                self.interactions.add((t_rc, action))

        # Recovery
        if self.stuck_counter > 2:
            action = self.ACT_USE if self.stuck_counter == 3 else self.ACT_INTERACT
            owner = "Recovery"

        self.last_action = action
        self.last_perspective = owner
        return action

    # --- PERSPECTIVES ---

    def _perspective_newton(self, grid, agent_rc, rows, cols):
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
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            target, color = ext
            return (0.92, self.ACT_CLICK, (target[0], target[1], color), "Euclid")
        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols):
        # Local Use/Click
        if agent_rc:
            adj = self._scan_adjacent(grid, agent_rc, rows, cols) + [agent_rc]
            for r, c in adj:
                val = grid[r, c]
                if val != 0:
                    # Bid Use
                    if ((r,c), self.ACT_USE) not in self.interactions:
                        score = 0.8 * GlobalNexus.CAUSAL_SCORES[val]
                        return (score, self.ACT_USE, (r,c), "Skinner")
                    # Bid Click (Fallback if Use failed)
                    if ((r,c), self.ACT_CLICK) not in self.interactions:
                        score = 0.75 * GlobalNexus.CAUSAL_SCORES[val]
                        return (score, self.ACT_CLICK, (r,c), "Skinner")

        # Global Click
        unique, counts = np.unique(grid, return_counts=True)
        for val in unique[np.argsort(counts)]:
            if val == 0: continue
            matches = np.argwhere(grid == val)
            for r, c in matches:
                if ((r, c), self.ACT_CLICK) not in self.interactions:
                     score = 0.6 * GlobalNexus.CAUSAL_SCORES[val]
                     return (score, self.ACT_CLICK, (r,c, val), "Skinner")

        return (0.0, None, None, "Skinner")

    def _perspective_fusion(self, grid, agent_rc, rows, cols):
        if not agent_rc: return (0.0, None, None, "Fusion")
        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            if dist > 1 and (t_rc, self.ACT_USE) not in self.interactions:
                val = grid[t_rc]
                adj = self._scan_adjacent(grid, t_rc, rows, cols)
                for near_rc in adj:
                    if grid[near_rc] == 0:
                        path = self._astar(grid, agent_rc, near_rc, rows, cols)
                        if path:
                            score = 0.94 * GlobalNexus.CAUSAL_SCORES[val]
                            return (score, path[0], t_rc, "Fusion")
        return (0.0, None, None, "Fusion")

    def _perspective_chaos(self):
        if self.control_map: return random.choice(list(self.control_map.keys()))
        return random.choice(self.RAW_ACTIONS)

    # --- SUBSTRATE & HANDSHAKE ---

    def _run_handshake(self, grid, agent_rc):
        # Did we learn anything?
        if self.last_pos and agent_rc and self.last_action in self.RAW_ACTIONS:
            dr, dc = agent_rc[0] - self.last_pos[0], agent_rc[1] - self.last_pos[1]
            if dr != 0 or dc != 0:
                self.control_map[self.last_action] = (dr, dc)
                GlobalNexus.CONTROLS[self.last_action] = (dr, dc)

        self.last_pos = agent_rc

        if self.handshake_queue:
            action = self.handshake_queue.popleft()
            self.last_action = action
            return action
        else:
            # QUEUE EMPTY
            if self.control_map:
                self.mode = "AVATAR"
            else:
                # CRITICAL FALLBACK: Assume we are AVATAR with Standard Controls
                # This fixes the "Default to Scientist" bug on ls20
                # Guess: 1=Up, 2=Down, 3=Left, 4=Right (Typical mappings vary but better than nothing)
                # We map to WASD-like: 1:(-1,0), 2:(1,0), 3:(0,-1), 4:(0,1)
                self.mode = "AVATAR"
                self.control_map = self.DEFAULT_CONTROLS.copy()
                GlobalNexus.CONTROLS = self.DEFAULT_CONTROLS.copy()

            return random.choice(self.RAW_ACTIONS)

    def _update_physics(self, grid, agent_rc, current_hash):
        if self.mode == "AVATAR" and self.last_grid is not None:
             if current_hash != np.sum(self.last_grid):
                 GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] *= 1.1

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

        # Check what appears to be the 'new' pixel
        cands = []
        diff_coords = np.argwhere(diff)
        for r, c in diff_coords:
            val = curr[r, c]
            prev_val = prev[r, c]
            # If it became non-zero, it might be the agent
            if val != 0 and prev_val == 0:
                cands.append(val)

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
