"""
NOVA-M-COP v1.6-AUTARKY
Sovereign Agent using Test-Time Training (TTT) & Active Inference.
- Learns Walls via Collision.
- Detects Goals via Rarity.
- Adapts via State Hashing (e.g., inventory/color changes).
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np  # Grid processing and uniqueness analysis
import random
from collections import deque

import numpy as np

from .agent import Agent
from .structs import FrameData, GameAction, GameState


class NovaAutarkyAgent(Agent):
    """
    NOVA-M-COP v1.6-AUTARKY
    Implements a fully autonomous O.O.D.A (Observe-Orient-Decide-Act) loop
    with test-time adaptation and no external dependencies.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Plan queue for committed multi-step paths
        self.plan: deque[GameAction] = deque()
        # Persistent knowledge base for learned environment features
        self.knowledge: dict[str, set[tuple[int, int]]] = {
            "walls": set(),       # Coordinates proven impassable via collision
            "bad_goals": set(),   # Targets proven unreachable (failed pathing)
        }
        self.last_pos: tuple[int, int] | None = None      # Previous agent position
        self.last_action: GameAction | None = None        # Last executed movement
        self.state_hash: int | None = None                # Hash of agent's own color/value (detects pickups/changes)
        self.stuck_count: int = 0                          # Consecutive failed movement attempts

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Terminal state detection — mirrors environment win/loss signals."""
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        """Core Autarkic O.O.D.A Loop — fully self-contained decision cycle."""
import numpy as np
import random
from collections import deque
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class NovaAutarkyAgent(Agent):
    """
    NOVA-M-COP v1.6-AUTARKY
    Sovereign Agent using Test-Time Training (TTT) & Active Inference.
    - Learns Walls via Collision.
    - Detects Goals via Rarity.
    - Adapts via State Hashing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plan = deque()
        self.knowledge = {
            "walls": set(),  # Coordinates of discovered walls
            "bad_goals": set(),  # Unreachable targets
        }
        self.last_pos = None
        self.last_action = None
        self.state_hash = None  # Tracks inventory/shape changes
            "walls": set(),        # Coordinates of discovered walls
            "bad_goals": set(),    # Unreachable targets
        }
        self.last_pos = None
        self.last_action = None
        self.state_hash = None     # Tracks inventory/shape changes
        self.stuck_count = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing or not."""
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        """The Autarkic O.O.D.A Loop"""
        if latest.state == GameState.NOT_PLAYED:
            return GameAction.RESET

        if not latest.frame:
            return GameAction.RESET

        # Parse the latest observation into a numpy grid (assumes last channel or 2D)
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape

        # Locate the agent in the current frame
        agent_pos = self._get_agent_pos(grid)

        # === 1. OBSERVE & LEARN (Collision-Based Test-Time Training) ===
        if self.last_pos and self.last_action:
            if self.last_pos == agent_pos:
                # Movement failed → collision detected
                self.stuck_count += 1
                # Compute intended destination tile
                dr, dc = self._action_to_delta(self.last_action)
                blocked_r, blocked_c = self.last_pos[0] + dr, self.last_pos[1] + dc
                # Learn: this tile is a permanent wall
                self.knowledge["walls"].add((blocked_r, blocked_c))

                # Severe stuck → abort current plan to force re-planning
                if self.stuck_count > 0:
                    self.plan.clear()
            else:
                # Successful movement → reset stuck counter
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape

        agent_pos = self._get_agent_pos(grid)

        # 1. PERCEPT & LEARN (L2 Ephemeral Memory)
        # Did our last action fail? (Collision Detection)
        if self.last_pos and self.last_action:
            if self.last_pos == agent_pos:
                self.stuck_count += 1
                # Calculate where we tried to go
                dr, dc = self._action_to_delta(self.last_action)
                blocked_r, blocked_c = self.last_pos[0] + dr, self.last_pos[1] + dc

                # LEARN: This tile is a wall.
                self.knowledge["walls"].add((blocked_r, blocked_c))

                # If stuck, abort current plan and re-think
                if self.stuck_count > 0:
                    self.plan.clear()
            else:
                self.stuck_count = 0

        self.last_pos = agent_pos

        # === 2. DETECT MAJOR STATE CHANGES (e.g., key pickup, color change) ===
        curr_hash = int(grid[agent_pos])  # Agent's own pixel value as proxy for state
        if self.state_hash is not None and curr_hash != self.state_hash:
            # Significant event occurred → clear plan and retry old goals
            self.plan.clear()
            self.knowledge["bad_goals"].clear()
        self.state_hash = curr_hash

        # === 3. DECIDE & ACT — Execute existing plan if available ===
        # 2. STATE CHECK (L1 Retina)
        # Did I change shape/color? (e.g., Picked up key)
        curr_hash = grid[agent_pos]
        if self.state_hash is not None and curr_hash != self.state_hash:
            # Major event! Clear plan to leverage new capabilities (e.g., unlocking door)
            self.plan.clear()
            self.knowledge["bad_goals"].clear()  # Retry previously failed goals
            self.knowledge["bad_goals"].clear() # Retry previously failed goals
        self.state_hash = curr_hash

        # 3. EXECUTE EXISTING PLAN
        if self.plan:
            action = self.plan.popleft()
            self.last_action = action
            return action

        # === 4. ORIENT — Infer goals via rarity heuristic ===
        targets = self._scan_for_targets(grid, agent_pos)

        best_path: list[GameAction] | None = None
        for target_pos in targets:
            if target_pos in self.knowledge["bad_goals"]:
                continue  # Skip previously unreachable targets

            # Attempt pathfinding with learned walls
            path = self._bfs(grid, agent_pos, target_pos)
            if path:
                best_path = path
                break  # Greedy: take first reachable target (closest due to sorting)
            else:
                # Mark as temporarily unreachable
                self.knowledge["bad_goals"].add(target_pos)

        # === 5. COMMIT TO PATH if found ===
        # 4. ORIENT (L3 Goal Inference)
        # Scan for objects. Filter out known walls and background (0).
        # Heuristic: Go to the nearest 'Unique' object.
        targets = self._scan_for_targets(grid, agent_pos)

        best_path = None
        for target_pos in targets:
            if target_pos in self.knowledge["bad_goals"]:
                continue
            if target_pos in self.knowledge["bad_goals"]: continue

            # Try to pathfind
            path = self._bfs(grid, agent_pos, target_pos)
            if path:
                best_path = path
                break
            else:
                # Mark as unreachable for now
                self.knowledge["bad_goals"].add(target_pos)

        # 5. ACT
        if best_path:
            self.plan.extend(best_path)
            action = self.plan.popleft()
            self.last_action = action
            return action

        # === 6. FALLBACK — Random exploration when no goals reachable ===
        action = random.choice(
            [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
        )
        self.last_action = action
        return action

    def _get_agent_pos(self, grid: np.ndarray) -> tuple[int, int]:
        """
        Robust agent localization heuristic.
        Prioritizes unique single-pixel objects (typical agent representation).
        Falls back to common agent colors.
        """
        unique, counts = np.unique(grid, return_counts=True)
        # Find colors appearing exactly once (strong agent signal)
        potential_agents = unique[(counts == 1) & (unique != 0)]

        if len(potential_agents) > 0:
            agent_color = potential_agents[0]
            coords = np.argwhere(grid == agent_color)
            if len(coords) > 0:
                return (int(coords[0][0]), int(coords[0][1]))

        # Fallback: known common agent colors (blue=1, red=2, cyan=8, etc.)
        for color in [1, 2, 8]:
            coords = np.argwhere(grid == color)
            if len(coords) == 1:
                return (int(coords[0][0]), int(coords[0][1]))

        # Ultimate fallback (should rarely trigger)
        return (0, 0)

    def _scan_for_targets(
        self, grid: np.ndarray, agent_pos: tuple[int, int]
    ) -> list[tuple[int, int]]:
        """Identify potential goals as rare non-background objects, sorted by Manhattan distance."""
        unique, counts = np.unique(grid, return_counts=True)
        # Rare = <15 pixels (keys, doors, switches, etc.)
        rare_colors = unique[(counts < 15) & (unique != 0)]

        candidates: list[tuple[int, tuple[int, int]]] = []
        # 6. FALLBACK (Explorer)
        # Random walk if no targets or trapped
        action = random.choice(
            [
                GameAction.ACTION1,
                GameAction.ACTION2,
                GameAction.ACTION3,
                GameAction.ACTION4,
            ]
        )
        action = random.choice([
            GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4
        ])
        self.last_action = action
        return action

    def _get_agent_pos(self, grid):
        """
        Heuristic to find the agent position.
        In ls20 and similar games, the agent is usually a unique colored pixel.
        """
        rows, cols = grid.shape
        # Priority search for single pixels of specific colors
        # Adjust these colors based on game knowledge if needed.
        # For now, just look for ANY single pixel color that isn't 0 (background).

        unique, counts = np.unique(grid, return_counts=True)
        # Filter for counts == 1 and value != 0
        potential_agents = unique[(counts == 1) & (unique != 0)]

        if len(potential_agents) > 0:
            # Pick the first one? Or prefer certain colors?
            # Let's pick the one that matches our previous hash if valid?
            # Or just the first one.
            agent_color = potential_agents[0]
            coords = np.argwhere(grid == agent_color)
            if len(coords) > 0:
                return tuple(coords[0])

        # If no single pixel found, fallback to searching for specific common agent colors
        # e.g. Blue (1), Red (2), Cyan (8)
        for color in [1, 2, 8]:
            coords = np.argwhere(grid == color)
            if len(coords) == 1:
                return tuple(coords[0])
             coords = np.argwhere(grid == color)
             if len(coords) == 1:
                 return tuple(coords[0])

        return (0, 0)

    def _scan_for_targets(self, grid, agent_pos):
        """Find rare objects, sorted by distance."""
        unique, counts = np.unique(grid, return_counts=True)
        # Rare objects are < 15 pixels (Keys, Doors, Switches)
        # Background is usually 0.
        rare_colors = unique[(counts < 15) & (unique != 0)]

        candidates = []
        rows, cols = grid.shape
        for r in range(rows):
            for c in range(cols):
                if grid[r, c] in rare_colors and (r, c) != agent_pos:
                    dist = abs(r - agent_pos[0]) + abs(c - agent_pos[1])
                    candidates.append((dist, (r, c)))

        candidates.sort()  # Closest first
        return [c[1] for c in candidates]

    def _bfs(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> list[GameAction] | None:
        """Breadth-first search respecting learned walls. Returns action sequence or None."""
        q: deque[tuple[tuple[int, int], list[GameAction]]] = deque([(start, [])])
        seen: set[tuple[int, int]] = {start}
        candidates.sort() # Closest first
        return [c[1] for c in candidates]

    def _bfs(self, grid, start, end):
        """Deterministic Pathfinding respecting Learned Walls."""
        q = deque([(start, [])])
        seen = {start}
        rows, cols = grid.shape

        while q:
            curr, path = q.popleft()
            if curr == end:
                return path

            if len(path) > 50:
                continue  # Depth limit for performance

            r, c = curr
            moves = [
                ((-1, 0), GameAction.ACTION1),  # Up
                ((1, 0), GameAction.ACTION2),   # Down
                ((0, -1), GameAction.ACTION3),  # Left
                ((0, 1), GameAction.ACTION4),   # Right
                continue  # Limit depth for speed
            if len(path) > 50: continue # Limit depth for speed

            r, c = curr
            moves = [
                ((-1, 0), GameAction.ACTION1),
                ((1, 0), GameAction.ACTION2),
                ((0, -1), GameAction.ACTION3),
                ((0, 1), GameAction.ACTION4),
                ((0, 1), GameAction.ACTION4)
            ]

            for (dr, dc), act in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Respect learned impassable tiles
                    if (nr, nc) in self.knowledge["walls"]:
                        continue

                    # Heuristic: walk on anything except known walls (allows pushing/interacting)
                    if (nr, nc) != end and grid[nr, nc] != 0:
                        pass  # Treat non-background as potentially traversable
                    # Check Learned Walls
                    if (nr, nc) in self.knowledge["walls"]:
                        continue
                    if (nr, nc) in self.knowledge["walls"]: continue

                    # Heuristic: Avoid stepping on other objects unless it's the target
                    # (Prevents getting stuck on non-goal clutter)
                    if (nr, nc) != end and grid[nr, nc] != 0:
                        # Treat unknown non-0 as potential obstacle until proven otherwise?
                        # For LS20, we treat them as walkable to interact.
                        pass
                         # Treat unknown non-0 as potential obstacle until proven otherwise?
                         # For LS20, we treat them as walkable to interact.
                         pass

                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path + [act]))
        return None

    def _action_to_delta(self, action: GameAction) -> tuple[int, int]:
        """Map movement actions to grid deltas for collision analysis."""
        deltas = {
            GameAction.ACTION1: (-1, 0),
            GameAction.ACTION2: (1, 0),
            GameAction.ACTION3: (0, -1),
            GameAction.ACTION4: (0, 1),
        }
        return deltas.get(action, (0, 0))
    def _action_to_delta(self, action):
        if action == GameAction.ACTION1:
            return (-1, 0)
        if action == GameAction.ACTION2:
            return (1, 0)
        if action == GameAction.ACTION3:
            return (0, -1)
        if action == GameAction.ACTION4:
            return (0, 1)
        if action == GameAction.ACTION1: return (-1, 0)
        if action == GameAction.ACTION2: return (1, 0)
        if action == GameAction.ACTION3: return (0, -1)
        if action == GameAction.ACTION4: return (0, 1)
        return (0, 0)
