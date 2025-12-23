import numpy as np
import random
from collections import deque
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class NovaOmniAgent(Agent):
    """
    NOVA-M-COP v1.7-OMNI
    Hybrid Agent: Auto-switches between 'Navigator' (Avatar) and 'Scientist' (God Mode).
    1. Calibration: Test physics (Do I have a body?).
    2. Avatar Mode: BFS + Push Logic (Sokoban/Nav).
    3. Scientist Mode: Probes interactions to find rule-based rewards.
    """

    # ACTION7 is typically the generic 'Interact/Use' (Spacebar) in ARC-AGI-3
    # Fallback to ACTION5 if ACTION7 is undefined in specific env version
    ACT_INTERACT = getattr(GameAction, 'ACTION7', GameAction.ACTION5)
    ACT_UP = GameAction.ACTION1 # Mapped from GameAction (1)
    ACT_DOWN = GameAction.ACTION2
    ACT_LEFT = GameAction.ACTION3
    ACT_RIGHT = GameAction.ACTION4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # State Tracking
        self.mode = "CALIBRATING"  # CALIBRATING -> AVATAR or SCIENTIST
        self.plan = deque()
        self.knowledge = {
            "walls": set(),
            "bad_goals": set(),
            "interactions": set()  # Track (pos) we have probed
        }
        self.last_grid = None     # Store numpy grid
        self.last_state = None    # (GridBytes, AgentPos)
        self.last_action = None
        self.stuck_count = 0
        self.calibration_steps = 0
        self.agent_color = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        """Main Agent Loop (Omni-Modal)"""
        if latest.state == GameState.NOT_PLAYED:
            return GameAction.RESET
        if not latest.frame:
            return GameAction.RESET

        grid = np.array(latest.frame[-1])
        # Robust Agent Position detection (Handles both Avatar and Cursor)
        agent_pos = self._get_agent_pos(grid)

        # --- L0: PHYSICS CLASSIFIER (Calibration Phase) ---
        if self.mode == "CALIBRATING":
            self.calibration_steps += 1

            # Step 1: Attempt to move to generate a delta
            if self.calibration_steps == 1:
                self.last_action = self.ACT_RIGHT
                self.last_state = (grid.tobytes(), agent_pos)
                self.last_grid = grid
                return self.ACT_RIGHT

            # Step 2: Analyze the result of Step 1
            prev_grid_bytes, prev_pos = self.last_state

            # Did we change position? (Avatar)
            pos_changed = (agent_pos != prev_pos) and (prev_pos is not None) and (agent_pos != (0,0))
            # Did the grid change visually? (e.g., cursor moved, or object pushed)
            grid_changed = (grid.tobytes() != prev_grid_bytes)

            # Refined Classification: Use Grid Diff to infer Agent Color if needed
            if grid_changed and not pos_changed:
                 # Grid changed but heuristic didn't track it. Heuristic is wrong.
                 # Identify what moved.
                 diff = grid != self.last_grid
                 changed_values = grid[diff]
                 candidates = [v for v in changed_values if v != 0]
                 if candidates:
                     from collections import Counter
                     most_common = Counter(candidates).most_common(1)
                     if most_common:
                         self.agent_color = most_common[0][0]
                         # Recalculate pos
                         agent_pos = self._get_agent_pos(grid)
                         pos_changed = (agent_pos != prev_pos)

            if pos_changed:
                # We have a physical body that moves.
                self.mode = "AVATAR"
            elif grid_changed:
                 # We didn't move bodily (or couldn't track it), but world changed.
                 # If we detected agent color above, we would be in AVATAR.
                 # If not, maybe it's a cursor? Or we failed to track.
                 # Default to AVATAR if we think it's a "body" movement (e.g. pushed something)?
                 # User logic says: "Did a specific pixel move? Avatar Mode."
                 # If grid changed, a pixel moved.
                 # So we should prefer AVATAR if we can track it.
                 # But sticking to user logic:
                 self.mode = "SCIENTIST"
            else:
                # Nothing happened. We might be blocked, or inputs are different.
                # Try INTERACT as a secondary probe.
                if self.calibration_steps == 2:
                    self.last_action = self.ACT_INTERACT
                    return self.ACT_INTERACT

                # If still static after Move+Interact, default to Scientist (God Mode)
                self.mode = "SCIENTIST"

        # --- L4: HYBRID ENGINE SWITCH ---
        if self.mode == "AVATAR":
            return self._run_avatar_logic(grid, agent_pos)
        else:
            return self._run_scientist_logic(grid, agent_pos)

    def _run_avatar_logic(self, grid, agent_pos):
        """v1.6 Autarky Navigation Logic + Interaction Upgrade"""
        if agent_pos is None or agent_pos == (0,0): return self._random_move()

        # 1. Learn Walls via Collision
        if self.last_action and self.last_state:
            # We need prev_pos from somewhere. self.last_state stores it but it might be stale if we didn't update it per step in loop.
            # We should update last_state every step.
            pass

        # We need to track last_state per step for Wall learning logic to work continously
        # The provided code updated last_state at end of logic.

        # Recover prev_pos logic
        # In this loop structure, we need to store current state for NEXT turn comparison.
        # But we also need PREVIOUS state for CURRENT comparison.
        # The user code does:
        #   if self.last_action and self.last_state: ...
        #   self.last_state = (grid.tobytes(), agent_pos)
        #   ... return action
        # This works.

        if self.last_action and self.last_state:
            prev_grid_bytes, prev_pos = self.last_state
            # Check for walls
            # If we tried to move and grid didn't change (collision)
            if self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
                 # Check if grid changed?
                 # User code uses: if prev_pos == agent_pos ...
                 # If agent_pos is robust, this works.
                 if prev_pos == agent_pos:
                     self.stuck_count += 1
                     if self.stuck_count > 1:
                         blocked_r, blocked_c = self._get_target_pos(prev_pos, self.last_action)
                         self.knowledge["walls"].add((blocked_r, blocked_c))
                         self.plan.clear()
                 else:
                     self.stuck_count = 0

        self.last_state = (grid.tobytes(), agent_pos)

        # 2. Execute Existing Plan
        if self.plan:
            action = self.plan.popleft()
            self.last_action = action
            return action

        # 3. Orient (Rare Objects Goal Inference)
        targets = self._scan_for_targets(grid, agent_pos)

        # 4. Pathfind
        for t in targets:
            if t in self.knowledge["bad_goals"]: continue
            path = self._bfs(grid, agent_pos, t, self.knowledge["walls"])
            if path:
                self.plan.extend(path)
                action = self.plan.popleft()
                self.last_action = action
                return action

        # 5. Fallback: Explore (Random)
        return self._random_move()

    def _run_scientist_logic(self, grid, agent_pos):
        """Causal Probe Logic for Non-Navigation Games (vc33, ft09)"""
        # If agent_pos is (0,0), it might be valid (top-left).

        # Heuristic: Find non-background objects and probe them.
        r, c = agent_pos

        # 1. Probe current location (Interaction)
        # If we are on a non-zero pixel we haven't touched, CLICK IT.
        if grid[r, c] != 0 and (r,c) not in self.knowledge["interactions"]:
             self.knowledge["interactions"].add((r,c))
             self.last_action = self.ACT_INTERACT
             return self.ACT_INTERACT

        # 2. Seek nearest interesting pixel
        targets = self._scan_for_targets(grid, agent_pos)
        valid_targets = [t for t in targets if t not in self.knowledge["interactions"]]

        if valid_targets:
            # Simple greedy step towards target
            tr, tc = valid_targets[0]
            if r < tr: return self.ACT_DOWN
            if r > tr: return self.ACT_UP
            if c < tc: return self.ACT_RIGHT
            if c > tc: return self.ACT_LEFT

        # 3. Random Search (Entropy)
        return self._random_move()

    def _get_agent_pos(self, grid):
        """Robustly find the agent or cursor."""
        # Use learned color if available
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0:
                 return tuple(coords[0])

        # Fallback: Scan for unique or rare pixels
        unique, counts = np.unique(grid, return_counts=True)

        # Sort by count
        sorted_indices = np.argsort(counts)
        sorted_colors = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]

        for color, count in zip(sorted_colors, sorted_counts):
            if color == 0 and count > 100: continue

            # Prefer unique (1)
            # Then rare (<50)
            if 0 < count < 50:
                coords = np.argwhere(grid == color)
                if len(coords) > 0:
                    return tuple(coords[0])

        return (0, 0)

    def _scan_for_targets(self, grid, agent_pos):
        """Rarity Heuristic: Find objects < 25 pixels."""
        if agent_pos is None: return []
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 25) & (unique != 0)]

        targets = []
        rows, cols = grid.shape
        # Optimized scanning
        matches = np.isin(grid, rare)
        coords = np.argwhere(matches)

        for r, c in coords:
            if (r, c) != agent_pos:
                dist = abs(r - agent_pos[0]) + abs(c - agent_pos[1])
                targets.append((dist, (r, c)))

        targets.sort() # Closest first
        return [t[1] for t in targets]

    def _bfs(self, grid, start, end, known_walls):
        """BFS Pathfinding respecting learned walls."""
        q = deque([(start, [])])
        seen = {start}
        rows, cols = grid.shape
        iters = 0
        while q:
            iters += 1
            if iters > 2000: return None # Timeout safety

            curr, path = q.popleft()
            if curr == end: return path
            if len(path) > 50: continue # Depth limit

            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in known_walls: continue
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path+[act]))
        return None

    def _get_target_pos(self, pos, action):
        r, c = pos
        if action == self.ACT_UP: return r-1, c
        if action == self.ACT_DOWN: return r+1, c
        if action == self.ACT_LEFT: return r, c-1
        if action == self.ACT_RIGHT: return r, c+1
        return r, c

    def _random_move(self):
        self.last_action = random.choice([self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT])
        return self.last_action
