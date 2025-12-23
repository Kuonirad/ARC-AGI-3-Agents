import random
from collections import deque

import numpy as np

from .agent import Agent
from .structs import FrameData, GameAction, GameState


import numpy as np
import random
from collections import deque
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class NovaPerceptronAgent(Agent):
    """
    NOVA-M-COP v1.8-PERCEPTRON
    Fixes:
    1. GEOMETRY: Maps API (x,y) -> Numpy [row,col] ([y,x]).
    2. NAVIGATION: 'Aggressive BFS' allows walking ON goals/keys.
    3. INTERACTION: Uses ACTION5 (Space) instead of ACTION7 (Undo?).
    """

    # Action Constants
    ACT_UP = GameAction.ACTION1
    ACT_DOWN = GameAction.ACTION2
    ACT_LEFT = GameAction.ACTION3
    ACT_RIGHT = GameAction.ACTION4
    ACT_INTERACT = getattr(GameAction, "ACTION5", GameAction.ACTION5)  # Space
    ACT_INTERACT = getattr(GameAction, 'ACTION5', GameAction.ACTION5) # Space

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = "CALIBRATING"
        self.plan = deque()
        self.knowledge = {
            "walls": set(),  # stored as (row, col)
            "bad_goals": set(),  # stored as (row, col)
            "interactions": set(),  # stored as (row, col)
        }
        self.last_state = None  # ((row,col), grid_bytes)
        self.last_action = None
        self.stuck_count = 0
        self.calibration_steps = 0
        self.inventory_hash = None  # Detects color changes (Key pickup)
            "walls": set(),        # stored as (row, col)
            "bad_goals": set(),    # stored as (row, col)
            "interactions": set()  # stored as (row, col)
        }
        self.last_state = None     # ((row,col), grid_bytes)
        self.last_action = None
        self.stuck_count = 0
        self.calibration_steps = 0
        self.inventory_hash = None # Detects color changes (Key pickup)
        self.agent_color = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        if latest.state == GameState.NOT_PLAYED:
            return GameAction.RESET
        if not latest.frame:
            return GameAction.RESET

        try:
            return self._safe_choose_action(latest)
        except Exception:
            return self._random_move()

    def _safe_choose_action(self, latest: FrameData) -> GameAction:
        grid = np.array(latest.frame[-1])
        # CRITICAL FIX: Coordinate Bridge
        # API gives (x,y). We convert to (row, col) -> (y, x).
        agent_rc = self._get_agent_rc(latest, grid)

        # --- L0: SMART CALIBRATION ---
        if self.mode == "CALIBRATING":
            self.calibration_steps += 1

            # Step 1: Find valid opening (Don't crash into wall)
            if self.calibration_steps == 1:
                move = self._find_open_move(grid, agent_rc)
                self.last_action = move
                self.last_state = (agent_rc, grid.tobytes())
                return move

            # Step 2: Classify Environment
            prev_rc, prev_bytes = self.last_state

            # Body Check: Did coordinates change?
            pos_changed = (agent_rc != prev_rc) and (prev_rc != (0, 0))

            # World Check: Did pixels change? (Cursor/God mode)
            grid_changed = grid.tobytes() != prev_bytes

            # Dynamic Color Logic (from v1.7)
            if grid_changed and not pos_changed:
                # Grid changed but heuristic didn't track it. Heuristic is wrong.
                # Identify what moved.
                last_grid = np.frombuffer(prev_bytes, dtype=grid.dtype).reshape(
                    grid.shape
                )
                diff = grid != last_grid
                changed_values = grid[diff]
                candidates = [v for v in changed_values if v != 0]
                if candidates:
                    from collections import Counter

                    most_common = Counter(candidates).most_common(1)
                    if most_common:
                        self.agent_color = most_common[0][0]
                        # Recalculate pos
                        agent_rc = self._get_agent_rc(latest, grid)
                        pos_changed = agent_rc != prev_rc

            if pos_changed:
                self.mode = "AVATAR"  # We have a body
            pos_changed = (agent_rc != prev_rc) and (prev_rc != (0,0))

            # World Check: Did pixels change? (Cursor/God mode)
            grid_changed = (grid.tobytes() != prev_bytes)

            # Dynamic Color Logic (from v1.7)
            if grid_changed and not pos_changed:
                 # Grid changed but heuristic didn't track it. Heuristic is wrong.
                 # Identify what moved.
                 last_grid = np.frombuffer(prev_bytes, dtype=grid.dtype).reshape(grid.shape)
                 diff = grid != last_grid
                 changed_values = grid[diff]
                 candidates = [v for v in changed_values if v != 0]
                 if candidates:
                     from collections import Counter
                     most_common = Counter(candidates).most_common(1)
                     if most_common:
                         self.agent_color = most_common[0][0]
                         # Recalculate pos
                         agent_rc = self._get_agent_rc(latest, grid)
                         pos_changed = (agent_rc != prev_rc)

            if pos_changed:
                self.mode = "AVATAR" # We have a body
            elif grid_changed:
                self.mode = "SCIENTIST"
            else:
                # Static? Try one interaction, then Scientist
                if self.calibration_steps < 3:
                    return self.ACT_INTERACT
                self.mode = "SCIENTIST"

        # --- EXECUTION ---
        if self.mode == "AVATAR":
            return self._run_avatar(grid, agent_rc)
        else:
            return self._run_scientist(grid, agent_rc)

    def _run_avatar(self, grid, agent_rc):
        """Logic for ls20 (Navigation)"""
        if not agent_rc or agent_rc == (0, 0):
            return self._random_move()

        # 1. Collision Learning
        if self.last_action in [
            self.ACT_UP,
            self.ACT_DOWN,
            self.ACT_LEFT,
            self.ACT_RIGHT,
        ]:
        if not agent_rc or agent_rc == (0,0): return self._random_move()

        # 1. Collision Learning
        if self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
            prev_rc = self.last_state[0] if self.last_state else None
            # Check grid diff for true collision detection
            last_grid_bytes = self.last_state[1] if self.last_state else None
            curr_grid_bytes = grid.tobytes()

            if last_grid_bytes == curr_grid_bytes:  # No change = Collision
            if last_grid_bytes == curr_grid_bytes: # No change = Collision
                self.stuck_count += 1
                if self.stuck_count > 1:
                    if prev_rc:
                        wr, wc = self._get_target_rc(prev_rc, self.last_action)
                        self.knowledge["walls"].add((wr, wc))
                        self.plan.clear()
            else:
                self.stuck_count = 0

        self.last_state = (agent_rc, grid.tobytes())

        # 2. Inventory Check (Key Pickup)
        curr_pixel = grid[agent_rc]
        if self.inventory_hash is not None and curr_pixel != self.inventory_hash:
            self.knowledge["bad_goals"].clear()
            self.plan.clear()
        self.inventory_hash = curr_pixel

        # 3. Execute Plan
        if self.plan:
            return self.plan.popleft()
        if self.plan: return self.plan.popleft()

        # 4. Aggressive Pathfinding
        # Find Rarest Non-Wall Objects
        targets = self._scan_targets(grid, agent_rc)
        for t in targets:
            if t in self.knowledge["bad_goals"]:
                continue
            if t in self.knowledge["bad_goals"]: continue
            path = self._bfs(grid, agent_rc, t, self.knowledge["walls"])
            if path:
                self.plan.extend(path)
                return self.plan.popleft()
            else:
                self.knowledge["bad_goals"].add(t)

        return self._random_move()

    def _run_scientist(self, grid, agent_rc):
        """Logic for vc33 (Cursor/God)"""
        # Strategy: Walk Cursor to Object -> INTERACT
        # If agent_rc is (0,0), it might be valid cursor pos.

        r, c = agent_rc

        # If ON object, Interact
        if grid[r, c] != 0 and (r, c) not in self.knowledge["interactions"]:
            self.knowledge["interactions"].add((r, c))
        if grid[r, c] != 0 and (r,c) not in self.knowledge["interactions"]:
            self.knowledge["interactions"].add((r,c))
            return self.ACT_INTERACT

        # Move to nearest new object
        targets = self._scan_targets(grid, agent_rc)
        valid = [t for t in targets if t not in self.knowledge["interactions"]]

        if valid:
            tr, tc = valid[0]
            if r < tr:
                return self.ACT_DOWN
            if r > tr:
                return self.ACT_UP
            if c < tc:
                return self.ACT_RIGHT
            if c > tc:
                return self.ACT_LEFT
            if r < tr: return self.ACT_DOWN
            if r > tr: return self.ACT_UP
            if c < tc: return self.ACT_RIGHT
            if c > tc: return self.ACT_LEFT

        return self._random_move()

    def _bfs(self, grid, start, end, walls):
        q = deque([(start, [])])
        seen = {start}
        rows, cols = grid.shape
        steps = 0
        while q:
            steps += 1
            if steps > 2000:
                break
            curr, path = q.popleft()
            if curr == end:
                return path
            if len(path) > 50:
                continue

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
                    if (nr, nc) in walls:
                        continue
            if steps > 2000: break
            curr, path = q.popleft()
            if curr == end: return path
            if len(path) > 50: continue

            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in walls: continue

                    # AGGRESSIVE BFS:
                    # Allow 0 (Air) OR Target (Key/Door)
                    val = grid[nr, nc]
                    is_walkable = (val == 0) or ((nr, nc) == end)

                    if is_walkable and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path + [act]))
                        q.append(((nr, nc), path+[act]))
        return None

    def _find_open_move(self, grid, rc):
        r, c = rc
        h, w = grid.shape
        # Prioritize Right, then others
        if c < w - 1 and grid[r, c + 1] == 0:
            return self.ACT_RIGHT
        if r < h - 1 and grid[r + 1, c] == 0:
            return self.ACT_DOWN
        if c > 0 and grid[r, c - 1] == 0:
            return self.ACT_LEFT
        if r > 0 and grid[r - 1, c] == 0:
            return self.ACT_UP
        if c < w-1 and grid[r, c+1] == 0: return self.ACT_RIGHT
        if r < h-1 and grid[r+1, c] == 0: return self.ACT_DOWN
        if c > 0 and grid[r, c-1] == 0: return self.ACT_LEFT
        if r > 0 and grid[r-1, c] == 0: return self.ACT_UP
        return self.ACT_RIGHT

    def _get_agent_rc(self, latest, grid):
        """Converts API (x,y) to Matrix (row, col)"""
        # Trust API (Safety Wrapper)
        if (
            hasattr(latest, "agent_pos")
            and latest.agent_pos
            and latest.agent_pos != (-1, -1)
        ):
            x, y = latest.agent_pos
            return (y, x)  # SWAP

        # Use learned color
        if self.agent_color is not None:
            coords = np.argwhere(grid == self.agent_color)
            if len(coords) > 0:
                return tuple(coords[0])
        if hasattr(latest, 'agent_pos') and latest.agent_pos and latest.agent_pos != (-1, -1):
            x, y = latest.agent_pos
            return (y, x) # SWAP

        # Use learned color
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0:
                 return tuple(coords[0])

        # Visual Scan (Robust)
        unique, counts = np.unique(grid, return_counts=True)
        sorted_indices = np.argsort(counts)
        sorted_colors = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]

        for color, count in zip(sorted_colors, sorted_counts):
            if color == 0 and count > 100:
                continue
            if 0 < count < 50:
                coords = np.argwhere(grid == color)
                if len(coords) > 0:
                    return tuple(coords[0])

        return (0, 0)  # Fallback
            if color == 0 and count > 100: continue
            if 0 < count < 50:
                coords = np.argwhere(grid == color)
                if len(coords) > 0: return tuple(coords[0])

        return (0, 0) # Fallback

    def _scan_targets(self, grid, agent_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 30) & (unique != 0)]
        matches = np.argwhere(np.isin(grid, rare))

        targets = []
        for r, c in matches:
            if (r, c) != agent_rc:
                d = abs(r - agent_rc[0]) + abs(c - agent_rc[1])
                targets.append((d, (r, c)))
        targets.sort()
        return [t[1] for t in targets]

    def _get_target_rc(self, rc, action):
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
        if action == self.ACT_UP: return r-1, c
        if action == self.ACT_DOWN: return r+1, c
        if action == self.ACT_LEFT: return r, c-1
        if action == self.ACT_RIGHT: return r, c+1
        return r, c

    def _random_move(self):
        return random.choice([self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT])
