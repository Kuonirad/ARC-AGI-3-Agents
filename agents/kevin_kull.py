import heapq
import os
import random
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- OPTIONAL VISION STACK (Hardware Agnostic / Free Energy Minimizer) ---
# Vision is OFF by default to avoid runtime stalls / model downloads.
# Enable with: NOVA_VISION=1 (and ensure model is available in local HF cache).
VISION_AVAILABLE = False
try:
    if os.environ.get("NOVA_VISION", "0") == "1":
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor

        VISION_AVAILABLE = True
except Exception:
    VISION_AVAILABLE = False


# --- SEMANTIC BRIDGE (Color Quantization) ---
COLOR_MAP = {
    "black": 0, "empty": 0, "background": 0,
    "blue": 1, "cyan": 1, "azure": 1,
    "red": 2, "crimson": 2, "maroon": 9, "burgundy": 9,
    "green": 3, "lime": 3, "forest": 3,
    "yellow": 4, "gold": 4, "lemon": 4,
    "grey": 5, "gray": 5, "silver": 5,
    "pink": 6, "magenta": 6, "purple": 6, "rose": 6,
    "orange": 7, "brown": 7, "amber": 7,
    "teal": 8, "turquoise": 8,
    "dark red": 9,
}

ARC_COLORS = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37)
]


def _resolve_action(name: str, fallback: GameAction) -> GameAction:
    """Safely resolve optional GameAction members without evaluating missing defaults."""
    try:
        return getattr(GameAction, name)
    except Exception:
        return fallback


# --- LOCAL VISION NEXUS (Coarse-Graining Module) ---
class LocalVisionNexus:
    _instance = None

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"

        if not VISION_AVAILABLE:
            return

        try:
            # Hardware Agnostic Loading: CUDA -> MPS -> CPU
            self._device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            dtype = torch.float16 if self._device != "cpu" else torch.float32

            # Florence-2 base (requires local cache if running offline)
            model_id = "microsoft/Florence-2-base"

            # If offline runners are used, loading can still fail; we keep it safe.
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=True,   # critical: don't try downloading
            ).to(self._device).eval()
            self._processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            self._model = None
            self._processor = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LocalVisionNexus()
        return cls._instance

    def analyze(self, grid_np: np.ndarray):
        """Return (tags, targets) where targets are [((r,c), label), ...]."""
        if self._model is None or self._processor is None:
            return [], []

        tags: List[str] = []
        targets: List[Tuple[Tuple[int, int], str]] = []

        try:
            h, w = grid_np.shape
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(10):
                img_rgb[grid_np == i] = ARC_COLORS[i]

            scale = max(1, 512 // max(h, w))
            pil_img = Image.fromarray(img_rgb).resize((w * scale, h * scale), Image.NEAREST)

            with torch.inference_mode():
                # Captioning
                inputs = self._processor(text="<MORE_DETAILED_CAPTION>", images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                if self._device == "mps" and "pixel_values" in inputs and inputs["pixel_values"].dtype != torch.float32:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

                gen_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values", None),
                    max_new_tokens=64,
                    num_beams=3,
                )
                text = self._processor.batch_decode(gen_ids, skip_special_tokens=False)[0].lower()

                if any(x in text for x in ["maze", "path", "navigate"]):
                    tags.append("NAV")
                if any(x in text for x in ["pattern", "grid", "fill"]):
                    tags.append("PATTERN")
                if any(x in text for x in ["button", "push", "switch", "object", "interact"]):
                    tags.append("INTERACT")

                # Object Detection
                inputs_od = self._processor(text="<OD>", images=pil_img, return_tensors="pt")
                inputs_od = {k: v.to(self._device) for k, v in inputs_od.items()}
                if self._device == "mps" and "pixel_values" in inputs_od:
                    inputs_od["pixel_values"] = inputs_od["pixel_values"].to(torch.float32)

                gen_od = self._model.generate(
                    input_ids=inputs_od["input_ids"],
                    pixel_values=inputs_od.get("pixel_values", None),
                    max_new_tokens=64,
                    num_beams=3,
                )
                text_od = self._processor.batch_decode(gen_od, skip_special_tokens=False)[0]

                res = self._processor.post_process_generation(
                    text_od, task="<OD>", image_size=(pil_img.width, pil_img.height)
                ).get("<OD>", {})

                for bbox, label in zip(res.get("bboxes", []), res.get("labels", [])):
                    cy = int(((bbox[1] + bbox[3]) / 2) / scale)
                    cx = int(((bbox[0] + bbox[2]) / 2) / scale)
                    if 0 <= cy < h and 0 <= cx < w:
                        targets.append(((cy, cx), str(label)))
        except Exception:
            pass

        return tags, targets


# --- GLOBAL NEXUS (Hyperion Upgrade) ---
class GlobalNexus:
    CONTROLS: Dict[GameAction, Tuple[int, int]] = {}

    # Free Energy Weights: Higher = Lower Surprise
    PERSPECTIVE_WEIGHTS = defaultdict(lambda: 1.0)

    # CAUSAL COMPRESSION GRAPH: (ObjectColor, ActionType) -> Probability of State Change
    CAUSAL_GRAPH = defaultdict(lambda: 0.5)

    VISION_CACHE: Dict[bytes, Any] = {}
    VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}

    @classmethod
    def clean_cache(cls):
        if len(cls.VISION_CACHE) > 50:
            cls.VISION_CACHE.clear()


class KevinKullAgent(Agent):
    """
    NOVA-M-COP v13.0-HYPERION
    Hierarchical Predictive Agent (Active Inference + Causal Compression).
    """

    MAX_ACTIONS = 400
    VISION_INTERVAL = 15

    RAW_ACTIONS = [
        GameAction.ACTION1,
        GameAction.ACTION2,
        GameAction.ACTION3,
        GameAction.ACTION4,
    ]

    DEFAULT_CONTROLS = {
        GameAction.ACTION1: (-1, 0),
        GameAction.ACTION2: (1, 0),
        GameAction.ACTION3: (0, -1),
        GameAction.ACTION4: (0, 1),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Resolve optional interaction actions safely
        self.ACT_USE = _resolve_action("ACTION5", GameAction.ACTION1)
        self.ACT_CLICK = _resolve_action("ACTION6", GameAction.ACTION2)
        self.ACT_INTERACT = _resolve_action("ACTION7", self.ACT_USE)
        self.INTERACTION_TYPES = [self.ACT_USE, self.ACT_CLICK, self.ACT_INTERACT]

        self.control_map: Dict[GameAction, Tuple[int, int]] = {}
        self.walls = set()
        self.bad_goals = set()
        self.interactions: Dict[Tuple[int, int], int] = {}
        self.visited = set()

        self.mode = "HANDSHAKE"
        self.handshake_queue = deque(self.RAW_ACTIONS)

        self.last_pos: Optional[Tuple[int, int]] = None
        self.last_grid: Optional[np.ndarray] = None
        self.last_action: Optional[GameAction] = None
        self.last_target_val: Optional[int] = None
        self.last_perspective: Optional[str] = None

        self.agent_color: Optional[int] = None
        self.bg_color = 0

        self.stuck_counter = 0
        self.lost_counter = 0
        self.step_count = 0

        self.action_data: Dict[str, Any] = {}

        self.vision_engine = LocalVisionNexus.get_instance() if VISION_AVAILABLE else None
        self.vision_targets: List[Tuple[Tuple[int, int], str]] = []

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in (GameState.WIN, GameState.GAME_OVER)

    def choose_action(self, frames: List[FrameData], latest: FrameData) -> GameAction:
        self.action_data = {}
        self.step_count += 1

        if not self.control_map and GlobalNexus.CONTROLS:
            self.control_map = GlobalNexus.CONTROLS.copy()
            self.mode = "AVATAR"

        if latest.state == GameState.NOT_PLAYED or latest.frame is None:
            return GameAction.RESET

        try:
            return self._hyperion_loop(latest)
        except Exception:
            return random.choice(self.RAW_ACTIONS)

    # --- CORE LOOP ---
    def _hyperion_loop(self, latest: FrameData) -> GameAction:
        grid = self._parse_grid(latest)
        rows, cols = grid.shape
        current_hash = grid.tobytes()

        # 0) Background inference
        vals, counts = np.unique(grid, return_counts=True)
        if len(vals) > 0:
            self.bg_color = int(vals[int(np.argmax(counts))])

        # 1) Vision oracle (optional)
        grid_changed = (self.last_grid is None) or (not np.array_equal(grid, self.last_grid))
        if self.vision_engine and (self.step_count % self.VISION_INTERVAL == 0 or grid_changed):
            GlobalNexus.clean_cache()
            if current_hash not in GlobalNexus.VISION_CACHE:
                tags, targets = self.vision_engine.analyze(grid)
                GlobalNexus.VISION_CACHE[current_hash] = (tags, targets)
            tags, targets = GlobalNexus.VISION_CACHE[current_hash]
            self.vision_targets = targets

            GlobalNexus.VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}
            if "NAV" in tags:
                GlobalNexus.VISION_HINTS["Newton"] = 1.5
            if "PATTERN" in tags:
                GlobalNexus.VISION_HINTS["Euclid"] = 1.5
            if "INTERACT" in tags:
                GlobalNexus.VISION_HINTS["Skinner"] = 1.5

        # 2) Locate agent
        agent_rc = self._locate_agent(latest, grid)
        if agent_rc is None and self.last_grid is not None and grid.shape == self.last_grid.shape:
            delta_rc = self._find_center_of_change(self.last_grid, grid)
            if delta_rc:
                agent_rc = delta_rc
                self.agent_color = int(grid[agent_rc])

        if agent_rc is not None:
            self.visited.add(agent_rc)
            self.lost_counter = 0
        else:
            self.lost_counter += 1

        # 3) Active inference learning update
        self._update_physics_and_learn(grid, agent_rc)

        if self.mode == "HANDSHAKE":
            return self._run_handshake(agent_rc)

        # 4) Parliament
        bids = []
        hints = GlobalNexus.VISION_HINTS

        # L0 Survival: blind pilot if we can't see avatar
        if self.mode == "AVATAR" and agent_rc is None:
            bid = 0.99 if self.lost_counter < 40 else 0.0
            bids.append((bid, self._perspective_chaos(), None, "BlindPilot"))

        # L1 Semantic oracle
        bids.append(self._perspective_oracle(grid))

        # L2 Physics (Newton + Fusion)
        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols, hints["Newton"]))
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        # L3 Abstract (Euclid + Skinner)
        bids.append(self._perspective_euclid(grid, rows, cols, hints["Euclid"]))
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols, hints["Skinner"]))

        # L4 Recovery
        bids.append(self._perspective_recovery())

        valid = [b for b in bids if b[1] is not None]
        if not valid:
            return self._perspective_chaos()

        # 5) Executive: Free Energy Minimization (score * reliability)
        weighted = []
        for score, action, target, owner in valid:
            w_score = float(score) * float(GlobalNexus.PERSPECTIVE_WEIGHTS[owner])

            # Veto: don't step into known walls when navigation actions are proposed
            if owner in ("Newton", "Fusion") and target and isinstance(target, tuple) and len(target) >= 2:
                if (target[0], target[1]) in self.walls:
                    w_score = 0.0

            weighted.append((w_score, action, target, owner))

        weighted.sort(key=lambda x: x[0], reverse=True)
        _, action, target, owner = weighted[0]

        # 6) Payload flooding only for interaction actions
        if target is not None and action in self.INTERACTION_TYPES:
            r, c = int(target[0]), int(target[1])
            val = None
            if len(target) >= 3:
                val = int(target[2])
            else:
                try:
                    gval = int(grid[r, c])
                    val = gval if gval != self.bg_color else 1
                except Exception:
                    val = 1

            self._set_payload((r, c), int(val))
            self.last_target_val = int(val)
            self.interactions[(r, c)] = self.step_count
        else:
            self.last_target_val = None
            self.action_data = {}

        self.last_action = action
        self.last_perspective = owner
        return action

    # --- PERSPECTIVES ---
    def _perspective_oracle(self, grid: np.ndarray):
        for (r, c), label in self.vision_targets:
            if (self.step_count - self.interactions.get((r, c), -100)) > 20:
                val = 1
                lab = str(label).lower()
                for name, v_int in COLOR_MAP.items():
                    if name in lab:
                        val = v_int
                        break
                return (0.97, self.ACT_CLICK, (int(r), int(c), int(val)), "Oracle")
        return (0.0, None, None, "Oracle")

    def _perspective_newton(self, grid, agent_rc, rows, cols, boost=1.0):
        if agent_rc is None:
            return (0.0, None, None, "Newton")

        targets = self._scan_targets(grid, agent_rc)
        for _, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals:
                continue
            path = self._astar(grid, agent_rc, t_rc, rows, cols, allow_push=True)
            if path:
                return (0.95 * boost, path[0], t_rc, "Newton")
            self.bad_goals.add(t_rc)

        frontier = self._find_frontier(agent_rc, rows, cols)
        if frontier:
            return (0.6 * boost, frontier[0], None, "Newton")

        return (0.0, None, None, "Newton")

    def _perspective_euclid(self, grid, rows, cols, boost=1.0):
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            (tr, tc), color = ext
            return (0.92 * boost, self.ACT_CLICK, (tr, tc, color), "Euclid")

        sym = self._detect_symmetry_completion(grid, rows, cols)
        if sym:
            (tr, tc), color = sym
            return (0.90 * boost, self.ACT_CLICK, (tr, tc, color), "Euclid")

        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols, boost=1.0):
        rot_action = self.INTERACTION_TYPES[self.step_count % len(self.INTERACTION_TYPES)]

        if agent_rc is not None:
            # Self interaction
            if (self.step_count - self.interactions.get(agent_rc, -100)) > 10:
                val = int(grid[agent_rc])
                if val != self.bg_color:
                    conf = GlobalNexus.CAUSAL_GRAPH[(val, rot_action)]
                    return (0.85 * boost * conf, rot_action, (agent_rc[0], agent_rc[1], val), "Skinner")

            # Adjacent interaction
            for r, c in self._scan_adjacent(agent_rc, rows, cols):
                val = int(grid[r, c])
                if val != self.bg_color and (self.step_count - self.interactions.get((r, c), -100)) > 10:
                    conf = GlobalNexus.CAUSAL_GRAPH[(val, rot_action)]
                    return (0.8 * boost * conf, rot_action, (r, c, val), "Skinner")

        # Global probing if avatar is unknown or we're "lost"
        if self.mode != "AVATAR" or self.lost_counter > 20:
            unique, counts = np.unique(grid, return_counts=True)
            order = np.argsort(counts)  # rare-first
            for idx in order:
                val = int(unique[idx])
                if val == self.bg_color:
                    continue
                matches = np.argwhere(grid == val)
                for r, c in matches:
                    r, c = int(r), int(c)
                    if (self.step_count - self.interactions.get((r, c), -100)) > 20:
                        conf = GlobalNexus.CAUSAL_GRAPH[(val, self.ACT_CLICK)]
                        return (0.6 * boost * conf, self.ACT_CLICK, (r, c, val), "Skinner")

        return (0.0, None, None, "Skinner")

    def _perspective_fusion(self, grid, agent_rc, rows, cols):
        if agent_rc is None:
            return (0.0, None, None, "Fusion")

        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            if dist <= 1:
                continue
            if (self.step_count - self.interactions.get(t_rc, -100)) <= 20:
                continue

            val = int(grid[t_rc])
            click_prob = GlobalNexus.CAUSAL_GRAPH[(val, self.ACT_CLICK)]
            use_prob = GlobalNexus.CAUSAL_GRAPH[(val, self.ACT_USE)]
            relevance = max(click_prob, use_prob)

            if relevance <= 0.4:
                continue

            for near_rc in self._scan_adjacent(t_rc, rows, cols):
                if int(grid[near_rc]) == self.bg_color and near_rc not in self.walls:
                    path = self._astar(grid, agent_rc, near_rc, rows, cols, allow_push=False)
                    if path:
                        # target is the interesting object, but we only move now
                        return (0.94 * relevance, path[0], t_rc, "Fusion")

        return (0.0, None, None, "Fusion")

    def _perspective_recovery(self):
        if self.stuck_counter > 2:
            if self.stuck_counter > 5:
                return (3.0, self._perspective_chaos(), None, "Recovery")
            action = self.INTERACTION_TYPES[self.stuck_counter % len(self.INTERACTION_TYPES)]
            return (0.98, action, None, "Recovery")
        return (0.0, None, None, "Recovery")

    def _perspective_chaos(self) -> GameAction:
        if self.control_map:
            return random.choice(list(self.control_map.keys()))
        return random.choice(self.RAW_ACTIONS)

    # --- SUBSTRATE & LEARNING ---
    def _run_handshake(self, agent_rc):
        # Infer control mapping from observed movement
        if self.last_pos and agent_rc and self.last_action in self.RAW_ACTIONS:
            dr = agent_rc[0] - self.last_pos[0]
            dc = agent_rc[1] - self.last_pos[1]
            if dr != 0 or dc != 0:
                self.control_map[self.last_action] = (dr, dc)
                GlobalNexus.CONTROLS[self.last_action] = (dr, dc)

        self.last_pos = agent_rc

        if self.handshake_queue:
            self.last_action = self.handshake_queue[0]
            return self.handshake_queue.popleft()

        # Switch into avatar mode
        self.mode = "AVATAR"
        if not self.control_map:
            self.control_map = self.DEFAULT_CONTROLS.copy()
        else:
            for k, v in self.DEFAULT_CONTROLS.items():
                if k not in self.control_map:
                    self.control_map[k] = v
        GlobalNexus.CONTROLS = self.control_map.copy()

        act = random.choice(self.RAW_ACTIONS)
        self.last_action = act
        return act

    def _update_physics_and_learn(self, grid: np.ndarray, agent_rc: Optional[Tuple[int, int]]):
        prev_grid = self.last_grid
        changed = (prev_grid is None) or (not np.array_equal(grid, prev_grid))
        moved = (agent_rc is not None and self.last_pos is not None and agent_rc != self.last_pos)

        # Update causal model from interaction outcomes
        if self.last_action is not None and prev_grid is not None:
            if self.last_action in self.INTERACTION_TYPES and self.last_target_val is not None:
                success = changed
                val = int(self.last_target_val)
                curr = float(GlobalNexus.CAUSAL_GRAPH[(val, self.last_action)])
                GlobalNexus.CAUSAL_GRAPH[(val, self.last_action)] = (
                    min(0.99, curr * 1.2) if success else max(0.1, curr * 0.8)
                )

        # Meta-cognition: reliability update
        if self.last_perspective:
            if moved or changed:
                GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] = min(
                    2.0, GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] * 1.05
                )
            elif self.stuck_counter > 2:
                GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] = max(
                    0.2, GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] * 0.95
                )

        if not changed and not moved:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # Wall inference if we tried to move but didn't
        if self.mode == "AVATAR" and self.last_pos and agent_rc == self.last_pos and self.last_action in self.control_map:
            dr, dc = self.control_map[self.last_action]
            tr, tc = self.last_pos[0] + dr, self.last_pos[1] + dc
            if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]:
                val = int(grid[tr, tc])

                is_interactive = any(
                    GlobalNexus.CAUSAL_GRAPH[(val, a)] > 0.6 for a in self.INTERACTION_TYPES
                )

                # Approx "wall" heuristic: huge regions of bg or very common tiles
                unique, counts = np.unique(grid, return_counts=True)
                count = int(counts[np.where(unique == val)][0]) if val in unique else 999

                if (count > 50 or val == self.bg_color) and not is_interactive:
                    self.walls.add((tr, tc))
                else:
                    # Encourage probing if not sure
                    self.interactions[(tr, tc)] = -100
                    self.stuck_counter = max(self.stuck_counter, 3)

                if self.stuck_counter > 8:
                    self.agent_color = None

        self.last_pos = agent_rc
        self.last_grid = grid.copy()

    # --- HELPERS ---
    def _set_payload(self, rc: Tuple[int, int], val: int):
        r, c = int(rc[0]), int(rc[1])
        payload = {
            "x": c, "y": r,
            "val": val, "value": val,
            "color": val, "symbol": val,
            "type": val, "target": val, "object": val,
            "destination": {"x": c, "y": r},
        }
        self.action_data = payload

        # Some runners read action_data off the action object itself; keep best-effort.
        for act in (self.ACT_CLICK, self.ACT_USE, self.ACT_INTERACT):
            try:
                act.action_data = payload
            except Exception:
                pass

    def _parse_grid(self, latest: FrameData) -> np.ndarray:
        raw = np.array(latest.frame)
        if raw.ndim == 2:
            return raw.astype(np.int32)
        if raw.ndim == 3:
            # common convention: last plane is latest
            return raw[-1].astype(np.int32)
        # last resort: try square-ish reshape
        size = raw.size
        side = int(np.sqrt(size))
        if side > 0 and side * side == size:
            return raw.reshape((side, side)).astype(np.int32)
        # fallback: 1-row
        return raw.reshape((1, -1)).astype(np.int32)

    def _astar(self, grid, start, end, rows, cols, allow_push=False):
        if not self.control_map:
            return None

        pq = [(0, 0, start, [])]  # (f, g, node, path)
        best_g = {start: 0}

        while pq:
            _, g, curr, path = heapq.heappop(pq)
            if curr == end:
                return path
            if g > 300:
                break

            r, c = curr
            for act, (dr, dc) in self.control_map.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in self.walls:
                    continue

                val = int(grid[nr, nc])
                is_walk = (val == self.bg_color) or ((nr, nc) == end)

                # Sokoban-ish lookahead: allow pushing a blocking tile if next is empty
                if allow_push and not is_walk:
                    nnr, nnc = nr + dr, nc + dc
                    if 0 <= nnr < rows and 0 <= nnc < cols:
                        if int(grid[nnr, nnc]) == self.bg_color and (nnr, nnc) not in self.walls:
                            is_walk = True

                if not is_walk:
                    continue

                step_cost = 1 if val == self.bg_color else 5
                if (nr, nc) in self.visited:
                    step_cost += 2

                ng = g + step_cost
                if (nr, nc) not in best_g or ng < best_g[(nr, nc)]:
                    best_g[(nr, nc)] = ng
                    h = abs(nr - end[0]) + abs(nc - end[1])
                    heapq.heappush(pq, (ng + h, ng, (nr, nc), path + [act]))

        return None

    def _locate_agent(self, latest: FrameData, grid: np.ndarray):
        # Runner-provided agent_pos is best if available
        try:
            if hasattr(latest, "agent_pos") and latest.agent_pos and latest.agent_pos != (-1, -1):
                # common: (x,y) -> (row,col)
                return (int(latest.agent_pos[1]), int(latest.agent_pos[0]))
        except Exception:
            pass

        if self.agent_color is not None:
            coords = np.argwhere(grid == self.agent_color)
            if len(coords) > 0:
                return (int(coords[0][0]), int(coords[0][1]))

        # Rarest non-bg tile heuristic
        unique, counts = np.unique(grid, return_counts=True)
        order = np.argsort(counts)
        for idx in order:
            c = int(unique[idx])
            if c == self.bg_color:
                continue
            coords = np.argwhere(grid == c)
            if len(coords) > 0:
                return (int(coords[0][0]), int(coords[0][1]))

        return None

    def _find_center_of_change(self, prev: np.ndarray, curr: np.ndarray):
        if prev.shape != curr.shape:
            return None
        diff = prev != curr
        coords = np.argwhere(diff)
        if len(coords) == 0:
            return None
        return (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))

    def _find_vector_extension(self, grid, rows, cols):
        non_bg = np.argwhere(grid != self.bg_color)
        if len(non_bg) > 100:
            return None

        by_color = defaultdict(list)
        for r, c in non_bg:
            by_color[int(grid[int(r), int(c)])].append((int(r), int(c)))

        for color, coords in by_color.items():
            if len(coords) < 2:
                continue
            coords.sort()
            for i in range(len(coords) - 1):
                r1, c1 = coords[i]
                r2, c2 = coords[i + 1]
                dr, dc = r2 - r1, c2 - c1
                if abs(dr) > 3 or abs(dc) > 3:
                    continue
                nr, nc = r2 + dr, c2 + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if int(grid[nr, nc]) == self.bg_color and (nr, nc) not in self.interactions:
                        return ((nr, nc), color)
        return None

    def _detect_symmetry_completion(self, grid, rows, cols):
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mc = cols - 1 - c
                val_l = int(grid[r, c])
                val_r = int(grid[r, mc])
                if val_l != self.bg_color and val_r == self.bg_color:
                    if (r, mc) not in self.interactions:
                        return ((r, mc), val_l)
                if val_r != self.bg_color and val_l == self.bg_color:
                    if (r, c) not in self.interactions:
                        return ((r, c), val_r)
        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 50) & (unique != self.bg_color)]
        matches = np.argwhere(np.isin(grid, rare))
        targets = []
        for r, c in matches:
            r, c = int(r), int(c)
            if (r, c) != start_rc:
                targets.append((abs(r - start_rc[0]) + abs(c - start_rc[1]), (r, c)))
        targets.sort()
        return targets

    def _scan_adjacent(self, rc, rows, cols):
        r, c = rc
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out.append((nr, nc))
        return out

    def _find_frontier(self, start, rows, cols):
        if not self.control_map:
            return None
        q = deque([(start, [])])
        seen = {start}
        expansions = 0

        while q:
            expansions += 1
            if expansions > 200:
                break
            curr, path = q.popleft()

            if len(path) > 8:
                return path

            r, c = curr
            for act, (dr, dc) in self.control_map.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if (nr, nc) in self.walls:
                    continue
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + [act]))

        return None
