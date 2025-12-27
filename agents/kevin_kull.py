import os
import random
import heapq
from collections import deque, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- OPTIONAL VISION STACK (Moondream2 Native) ---
# Enable via NOVA_VISION=1 env var.
VISION_AVAILABLE = False
try:
    if os.environ.get("NOVA_VISION", "0") == "1":
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM
        VISION_AVAILABLE = True
except Exception:
    VISION_AVAILABLE = False


# --- SEMANTIC BRIDGE ---
COLOR_MAP = {
    "black": 0, "empty": 0, "background": 0,
    "blue": 1, "cyan": 1, "azure": 1,
    "red": 2, "crimson": 2, "maroon": 9, "burgundy": 9,
    "green": 3, "lime": 3, "forest": 3,
    "yellow": 4, "gold": 4, "lemon": 4,
    "grey": 5, "gray": 5, "silver": 5,
    "pink": 6, "magenta": 6, "purple": 6, "rose": 6,
    "orange": 7, "brown": 7, "amber": 7,
    "teal": 8, "turquoise": 8, "dark red": 9,
}

ARC_COLORS = [
    (0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
    (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37)
]


def _resolve_action(name: str, fallback: GameAction) -> GameAction:
    try:
        return getattr(GameAction, name)
    except Exception:
        return fallback


# --- LOCAL VISION NEXUS (Moondream2) ---
class LocalVisionNexus:
    _instance = None

    def __init__(self):
        self._model = None
        if not VISION_AVAILABLE:
            return
        try:
            self._device = (
                "cuda" if torch.cuda.is_available()
                else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
            )
            dtype = torch.float16 if self._device != "cpu" else torch.float32

            # Moondream2 (Revision 2025-06-21)
            model_id = "vikhyatk/moondream2"
            revision = "2025-06-21"

            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=True
            ).to(self._device).eval()

        except Exception:
            self._model = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LocalVisionNexus()
        return cls._instance

    def analyze(self, grid_np):
        if not self._model:
            return [], []

        tags, targets = [], []
        try:
            h, w = grid_np.shape
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(10):
                img_rgb[grid_np == i] = ARC_COLORS[i]

            scale = max(1, 384 // max(h, w))
            pil_img = Image.fromarray(img_rgb).resize((w * scale, h * scale), Image.NEAREST)

            with torch.inference_mode():
                try:
                    cap = self._model.caption(pil_img, length="normal")
                    text = cap["caption"].lower()

                    if any(x in text for x in ["maze", "path", "navigate", "move"]):
                        tags.append("NAV")
                    if any(x in text for x in ["pattern", "fill", "complete", "symmetry", "sequence"]):
                        tags.append("PATTERN")
                    if any(x in text for x in ["button", "push", "switch", "interact", "press", "toggle"]):
                        tags.append("INTERACT")
                except:
                    pass

                try:
                    det = self._model.detect(pil_img, "objects")
                    for obj in det.get("objects", []):
                        bbox = obj.get("bbox", [0, 0, 1, 1])
                        label = obj.get("label", "unknown").lower().strip()

                        cx = int(((bbox[0] + bbox[2]) / 2) * w)
                        cy = int(((bbox[1] + bbox[3]) / 2) * h)

                        if 0 <= cy < h and 0 <= cx < w:
                            targets.append(((cy, cx), label))
                except:
                    pass
        except Exception:
            pass
        return tags, targets


# --- GLOBAL NEXUS (Composite Stigmergy) ---
class GlobalNexus:
    CONTROLS: Dict[GameAction, Tuple[int, int]] = {}
    PERSPECTIVE_WEIGHTS = defaultdict(lambda: 1.0)

    USE_SCORES = defaultdict(lambda: 1.0)
    CLICK_SCORES = defaultdict(lambda: 1.0)

    CAUSAL_GRAPH = defaultdict(lambda: 0.5)

    STIGMERGIC_GRAPH = defaultdict(lambda: [0.0, 0.0, 0])

    VISION_CACHE = {}
    VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}

    GLOBAL_STEP = 0

    @classmethod
    def clean_cache(cls):
        if len(cls.VISION_CACHE) > 50:
            cls.VISION_CACHE.clear()

    @staticmethod
    def _coord_hash(r: int, c: int) -> int:
        return (int(r) << 16) | int(c)

    @classmethod
    def stig_update(cls, composite_key: str, action: GameAction, r: int, c: int, success: bool, current_step: int):
        key = (composite_key, action, cls._coord_hash(r, c))
        succ, att, last_step = cls.STIGMERGIC_GRAPH[key]

        steps_elapsed = max(0, current_step - last_step)
        decay = 0.995 ** steps_elapsed
        succ *= decay
        att *= decay

        att += 1.0
        if success:
            succ += 1.0

        cls.STIGMERGIC_GRAPH[key] = [succ, att, current_step]

        if "_" in composite_key:
            try:
                base_val = int(composite_key.split("_")[-1])
                current_legacy = cls.CAUSAL_GRAPH[(base_val, action)]
                new_legacy = succ / att if att > 0 else 0.5
                cls.CAUSAL_GRAPH[(base_val, action)] = 0.9 * current_legacy + 0.1 * new_legacy
            except: pass

    @classmethod
    def stig_query(cls, composite_key: str, action: GameAction, r: int, c: int, current_step: int) -> float:
        key = (composite_key, action, cls._coord_hash(r, c))
        succ, att, last_step = cls.STIGMERGIC_GRAPH[key]

        if att < 0.1:
            if "_" in composite_key:
                try:
                    val = int(composite_key.split("_")[-1])
                    return cls.CAUSAL_GRAPH[(val, action)]
                except: pass
            return 0.5

        steps_elapsed = max(0, current_step - last_step)
        decay = 0.995 ** steps_elapsed

        d_succ = succ * decay
        d_att = att * decay

        if d_att < 0.1: return 0.5
        return min(0.99, max(0.01, d_succ / d_att))


class KevinKullAgent(Agent):
    """
    NOVA-M-COP v15.0-COMPOSITE (KEVIN_KULL)
    The 'Semantic Solvency' Agent.
    """
    MAX_ACTIONS = 400
    VISION_INTERVAL = 15

    RAW_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    DEFAULT_CONTROLS = {
        GameAction.ACTION1: (-1, 0), GameAction.ACTION2: (1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ACT_USE = _resolve_action("ACTION5", GameAction.ACTION1)
        self.ACT_CLICK = _resolve_action("ACTION6", GameAction.ACTION2)
        self.ACT_INTERACT = _resolve_action("ACTION7", self.ACT_USE)
        self.INTERACTION_TYPES = [self.ACT_USE, self.ACT_CLICK, self.ACT_INTERACT]

        self.control_map = {}
        self.walls = set()
        self.bad_goals = set()
        self.interactions = {}
        self.collision_memory = {}
        self.visited = set()
        self.mode = "HANDSHAKE"
        self.handshake_queue = deque(self.RAW_ACTIONS)

        self.last_pos = None
        self.last_grid = None
        self.last_action = None
        self.last_target_val = None
        self.last_nav_target = None
        self.last_perspective = None

        self.agent_color = None
        self.bg_color = 0
        self.rare_vals = set()

        self.stuck_counter = 0
        self.lost_counter = 0
        self.step_count = 0
        self.action_data = {}
        self.vision_engine = LocalVisionNexus.get_instance() if VISION_AVAILABLE else None
        self.vision_targets = []

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        done = latest_frame.state in (GameState.WIN, GameState.GAME_OVER)
        if done and latest_frame.state == GameState.WIN and self.last_target_val is not None:
             GlobalNexus.CAUSAL_GRAPH[(self.last_target_val, self.last_action)] *= 10.0
        return done

    def choose_action(self, frames: List[FrameData], latest: FrameData) -> GameAction:
        self.action_data = {}
        self.step_count += 1
        GlobalNexus.GLOBAL_STEP += 1

        if not self.control_map and GlobalNexus.CONTROLS:
            self.control_map = GlobalNexus.CONTROLS.copy()
            self.mode = "AVATAR"
        if latest.state == GameState.NOT_PLAYED or latest.frame is None: return GameAction.RESET
        try: return self._composite_loop(latest)
        except: return random.choice(self.RAW_ACTIONS)

    def _composite_loop(self, latest: FrameData) -> GameAction:
        grid = self._parse_grid(latest)
        rows, cols = grid.shape
        current_hash = grid.tobytes()

        prev_pos = self.last_pos
        prev_action = self.last_action

        vals, counts = np.unique(grid, return_counts=True)
        if len(vals) > 0:
            self.bg_color = int(vals[int(np.argmax(counts))])
            self.rare_vals = set(vals[counts < 60].astype(int))

        grid_changed = (self.last_grid is None) or (current_hash != self.last_grid.tobytes())
        if self.vision_engine and (self.step_count % self.VISION_INTERVAL == 0 or grid_changed):
            GlobalNexus.clean_cache()
            if current_hash not in GlobalNexus.VISION_CACHE:
                tags, targets = self.vision_engine.analyze(grid)
                GlobalNexus.VISION_CACHE[current_hash] = (tags, targets)
            tags, targets = GlobalNexus.VISION_CACHE[current_hash]
            self.vision_targets = targets

            GlobalNexus.VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}
            if "NAV" in tags: GlobalNexus.VISION_HINTS["Newton"] = 1.5
            if "PATTERN" in tags: GlobalNexus.VISION_HINTS["Euclid"] = 1.5
            if "INTERACT" in tags: GlobalNexus.VISION_HINTS["Skinner"] = 1.5

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

        self._update_physics_and_learn(grid, agent_rc)

        if self.mode == "HANDSHAKE":
            return self._run_handshake(agent_rc, prev_pos, prev_action)

        bids = []
        hints = GlobalNexus.VISION_HINTS

        if self.mode == "AVATAR" and agent_rc is None:
            bid = 0.99 if self.lost_counter < 40 else 0.0
            bids.append((bid, self._perspective_chaos(), None, "BlindPilot"))

        bids.append(self._perspective_oracle(grid))

        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols, hints["Newton"]))
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        bids.append(self._perspective_euclid(grid, rows, cols, hints["Euclid"]))
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols, hints["Skinner"]))
        bids.append(self._perspective_recovery())

        valid = [b for b in bids if b[1] is not None]
        if not valid: return self._perspective_chaos()

        weighted = []
        for score, action, target, owner in valid:
            w_score = float(score) * float(GlobalNexus.PERSPECTIVE_WEIGHTS[owner])
            if owner in ("Newton", "Fusion") and target:
                t_rc = (int(target[0]), int(target[1]))
                if t_rc in self.walls: w_score *= 0.0
                elif (self.step_count - self.collision_memory.get(t_rc, -100) < 50): w_score *= 0.0
            weighted.append((w_score, action, target, owner))

        weighted.sort(key=lambda x: x[0], reverse=True)
        _, action, target, owner = weighted[0]

        if target:
            r, c = int(target[0]), int(target[1])
            val = 1
            if len(target) >= 3:
                val = int(target[2])
            else:
                try:
                    gval = int(grid[r, c])
                    val = gval if gval != self.bg_color else 1
                except: pass
            if val == 0 and owner != "Euclid": val = 1

            self._set_payload((r, c), val)
            self.last_target_val = val
            self.last_nav_target = (r, c)
            if action in self.INTERACTION_TYPES or action == self.ACT_CLICK:
                self.interactions[(r, c)] = self.step_count
        else:
            self.last_target_val = None
            self.last_nav_target = None

        self.last_action = action
        self.last_perspective = owner
        return action

    def _perspective_oracle(self, grid):
        for (r, c), label in self.vision_targets:
            r, c = int(r), int(c)
            if (self.step_count - self.interactions.get((r, c), -100)) > 20:
                val = 1
                low = str(label).lower()
                for name, v_int in COLOR_MAP.items():
                    if name in low: val = v_int; break
                return (0.97, self.ACT_CLICK, (r, c, int(val)), f"Oracle({label})")
        return (0.0, None, None, "Oracle")

    def _perspective_newton(self, grid, agent_rc, rows, cols, boost=1.0):
        if not agent_rc: return (0.0, None, None, "Newton")
        targets = self._scan_targets(grid, agent_rc)
        for _, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue
            if (self.step_count - self.collision_memory.get(t_rc, -100) < 50): continue
            path = self._astar(grid, agent_rc, t_rc, rows, cols, allow_push=True)
            if path: return (0.95 * boost, path[0], t_rc, "Newton")
            self.bad_goals.add(t_rc)
        frontier = self._find_frontier(agent_rc, rows, cols)
        if frontier: return (0.6 * boost, frontier[0], None, "Newton")
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

        def _lookup_label(r, c):
            for (vr, vc), label in self.vision_targets:
                if (vr, vc) == (r, c): return label
            return "unknown"

        if agent_rc is not None:
            if (self.step_count - self.interactions.get(agent_rc, -100) > 10):
                val = int(grid[agent_rc])
                if val != self.bg_color:
                    comp = f"{_lookup_label(agent_rc[0], agent_rc[1])}_{val}"
                    conf = GlobalNexus.stig_query(comp, rot_action, agent_rc[0], agent_rc[1], GlobalNexus.GLOBAL_STEP)
                    return (0.85 * boost * conf, rot_action, (agent_rc[0], agent_rc[1], val), "Skinner")
            for r, c in self._scan_adjacent(agent_rc, rows, cols):
                val = int(grid[r, c])
                if val != self.bg_color and (self.step_count - self.interactions.get((r, c), -100)) > 10:
                    comp = f"{_lookup_label(r, c)}_{val}"
                    conf = GlobalNexus.stig_query(comp, rot_action, r, c, GlobalNexus.GLOBAL_STEP)
                    return (0.8 * boost * conf, rot_action, (r, c, val), "Skinner")
        if self.mode != "AVATAR" or self.lost_counter > 20:
            unique, counts = np.unique(grid, return_counts=True)
            for val in unique[np.argsort(counts)]:
                val = int(val)
                if val == self.bg_color: continue
                matches = np.argwhere(grid == val)
                for r, c in matches:
                    r, c = int(r), int(c)
                    if (self.step_count - self.interactions.get((r, c), -100)) > 20:
                        comp = f"{_lookup_label(r, c)}_{val}"
                        conf = GlobalNexus.stig_query(comp, self.ACT_CLICK, r, c, GlobalNexus.GLOBAL_STEP)
                        return (0.6 * boost * conf, self.ACT_CLICK, (r, c, val), "Skinner")
        return (0.0, None, None, "Skinner")

    def _perspective_fusion(self, grid, agent_rc, rows, cols):
        if not agent_rc: return (0.0, None, None, "Fusion")
        targets = self._scan_targets(grid, agent_rc)

        def _lookup_label(r, c):
            for (vr, vc), label in self.vision_targets:
                if (vr, vc) == (r, c): return label
            return "unknown"

        for dist, t_rc in targets:
            if dist <= 1: continue
            if (self.step_count - self.interactions.get(t_rc, -100) <= 20): continue
            if (self.step_count - self.collision_memory.get(t_rc, -100) < 50): continue

            val = int(grid[t_rc])
            comp = f"{_lookup_label(t_rc[0], t_rc[1])}_{val}"

            click_conf = GlobalNexus.stig_query(comp, self.ACT_CLICK, t_rc[0], t_rc[1], GlobalNexus.GLOBAL_STEP)
            use_conf = GlobalNexus.stig_query(comp, self.ACT_USE, t_rc[0], t_rc[1], GlobalNexus.GLOBAL_STEP)
            rel = max(click_conf, use_conf)

            if rel <= 0.4: continue
            for near_rc in self._scan_adjacent(t_rc, rows, cols):
                if int(grid[near_rc]) == self.bg_color:
                    path = self._astar(grid, agent_rc, near_rc, rows, cols, allow_push=False)
                    if path: return (0.94 * rel, path[0], t_rc, "Fusion")
        return (0.0, None, None, "Fusion")

    def _perspective_recovery(self):
        if self.stuck_counter > 2:
            if self.stuck_counter > 5: return (3.0, self._perspective_chaos(), None, "Recovery")
            action = self.INTERACTION_TYPES[self.stuck_counter % len(self.INTERACTION_TYPES)]
            return (0.98, action, None, "Recovery")
        return (0.0, None, None, "Recovery")

    def _perspective_chaos(self):
        if self.control_map: return random.choice(list(self.control_map.keys()))
        return random.choice(self.RAW_ACTIONS)

    def _run_handshake(self, agent_rc, prev_pos, prev_action):
        if prev_pos and agent_rc and prev_action in self.RAW_ACTIONS:
            dr = int(agent_rc[0] - prev_pos[0]); dc = int(agent_rc[1] - prev_pos[1])
            if dr != 0 or dc != 0:
                self.control_map[prev_action] = (dr, dc)
                GlobalNexus.CONTROLS[prev_action] = (dr, dc)
        if self.handshake_queue:
            self.last_action = self.handshake_queue[0]
            return self.handshake_queue.popleft()
        self.mode = "AVATAR"
        if not self.control_map: self.control_map = self.DEFAULT_CONTROLS.copy()
        else:
            for k, v in self.DEFAULT_CONTROLS.items():
                if k not in self.control_map: self.control_map[k] = v
        GlobalNexus.CONTROLS = self.control_map.copy()
        act = random.choice(self.RAW_ACTIONS)
        self.last_action = act
        return act

    def _update_physics_and_learn(self, grid, agent_rc):
        current_hash = grid.tobytes()
        prev_hash = self.last_grid.tobytes() if self.last_grid is not None else b""
        changed = (current_hash != prev_hash)
        moved = (agent_rc is not None and self.last_pos is not None and agent_rc != self.last_pos)
        success = changed or moved

        if changed:
            self.bad_goals.clear()
            self.collision_memory.clear()
            self.walls.clear()

        if self.last_action in (self.INTERACTION_TYPES + [self.ACT_CLICK]):
             if self.last_nav_target or agent_rc:
                 r, c = self.last_nav_target if self.last_nav_target else agent_rc
                 val = self.last_target_val if self.last_target_val is not None else 0

                 label = "unknown"
                 for (vr, vc), vl in self.vision_targets:
                     if (vr, vc) == (r, c):
                         label = vl.lower().strip(); break

                 comp = f"{label}_{val}"
                 GlobalNexus.stig_update(comp, self.last_action, r, c, success, GlobalNexus.GLOBAL_STEP)

        if self.last_perspective:
            if success: GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] = min(2.0, GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] * 1.05)
            elif self.stuck_counter > 2: GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] = max(0.2, GlobalNexus.PERSPECTIVE_WEIGHTS[self.last_perspective] * 0.95)

        if not success:
            self.stuck_counter += 1
            if self.stuck_counter > 2 and self.last_nav_target:
                self.collision_memory[self.last_nav_target] = self.step_count
        else:
            self.stuck_counter = 0

        if self.mode == "AVATAR" and self.stuck_counter > 12:
            self.control_map = {}
            GlobalNexus.CONTROLS = {}
            self.mode = "HANDSHAKE"
            self.handshake_queue = deque(self.RAW_ACTIONS)
            self.stuck_counter = 0

        if self.mode == "AVATAR" and self.last_pos and agent_rc == self.last_pos and self.last_action in self.control_map:
            dr, dc = self.control_map[self.last_action]
            tr, tc = self.last_pos[0] + dr, self.last_pos[1] + dc
            if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1]:
                val = int(grid[tr, tc])
                label = "unknown"
                for (vr, vc), vl in self.vision_targets:
                    if (vr, vc) == (tr, tc): label = vl.lower().strip(); break
                comp = f"{label}_{val}"

                is_interactive = False
                for a in self.INTERACTION_TYPES:
                     if GlobalNexus.stig_query(comp, a, tr, tc, GlobalNexus.GLOBAL_STEP) > 0.6:
                         is_interactive = True; break

                unique, counts = np.unique(grid, return_counts=True)
                count = int(counts[np.where(unique == val)][0]) if val in unique else 999
                if (count > 50 or val == self.bg_color) and not is_interactive:
                    self.walls.add((tr, tc))
                else:
                    self.interactions[(tr, tc)] = -100
                    self.stuck_counter = max(self.stuck_counter, 3)
                if self.stuck_counter > 8: self.agent_color = None

        if self.step_count % 250 == 0:
            cutoff = self.step_count - 800
            self.collision_memory = {k: v for k, v in self.collision_memory.items() if v >= cutoff}
            self.interactions = {k: v for k, v in self.interactions.items() if (v < 0) or (v >= cutoff)}
            if len(self.visited) > 5000: self.visited.clear()

        self.last_pos = agent_rc
        self.last_grid = grid.copy()

    def _set_payload(self, rc, val):
        r, c = int(rc[0]), int(rc[1])
        payload = {
            "x": c, "y": r, "r": r, "c": c, "row": r, "col": c,
            "val": val, "value": val, "color": val, "symbol": val,
            "type": val, "target": val, "object": val,
            "destination": {"x": c, "y": r, "r": r, "c": c, "row": r, "col": c},
        }
        self.action_data = payload
        for act in (self.ACT_CLICK, self.ACT_USE, self.ACT_INTERACT):
            try:
                if hasattr(act, 'set_data'): act.set_data(payload)
                else: act.action_data = payload
            except: pass

    def _parse_grid(self, latest):
        raw = np.array(latest.frame)
        if raw.ndim == 2: return raw.astype(np.int32)
        if raw.ndim == 3: return raw[-1].astype(np.int32)
        return raw.reshape((int(np.sqrt(raw.size)), -1)).astype(np.int32)

    def _astar(self, grid, start, end, rows, cols, allow_push=False):
        if not self.control_map: return None
        pq = [(0, 0, start, [])]; best_g = {start: 0}
        while pq:
            _, g, curr, path = heapq.heappop(pq)
            if curr == end: return path
            if g > 300: break
            r, c = curr
            for act, (dr, dc) in self.control_map.items():
                nr, nc = r+dr, c+dc
                if not (0 <= nr < rows and 0 <= nc < cols): continue
                if (nr, nc) in self.walls: continue
                val = int(grid[nr, nc])
                is_walk = (val == self.bg_color) or ((nr, nc) == end) or (val in self.rare_vals)
                if allow_push and not is_walk:
                    nnr, nnc = nr+dr, nc+dc
                    if 0 <= nnr < rows and 0 <= nnc < cols:
                        if int(grid[nnr, nnc]) == self.bg_color and (nnr, nnc) not in self.walls: is_walk = True
                if is_walk:
                    cost = 1 if val == self.bg_color else 20
                    if (nr, nc) in self.visited: cost += 2
                    ng = g + cost
                    if (nr, nc) not in best_g or ng < best_g[(nr, nc)]:
                        best_g[(nr, nc)] = ng
                        heapq.heappush(pq, (ng+abs(nr-end[0])+abs(nc-end[1]), ng, (nr, nc), path+[act]))
        return None

    def _locate_agent(self, latest, grid):
        try:
            if hasattr(latest, "agent_pos") and latest.agent_pos and latest.agent_pos != (-1, -1):
                return (int(latest.agent_pos[1]), int(latest.agent_pos[0]))
        except: pass
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0: return (int(coords[0][0]), int(coords[0][1]))
        unique, counts = np.unique(grid, return_counts=True)
        order = np.argsort(counts)
        for idx in order:
            c = int(unique[idx])
            if c == self.bg_color: continue
            coords = np.argwhere(grid == c)
            if len(coords) > 0: return (int(coords[0][0]), int(coords[0][1]))
        return None

    def _find_center_of_change(self, prev, curr):
        if prev.shape != curr.shape: return None
        diff = prev != curr
        coords = np.argwhere(diff)
        if len(coords) == 0: return None
        return (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))

    def _find_vector_extension(self, grid, rows, cols):
        non_bg = np.argwhere(grid != self.bg_color)
        if len(non_bg) > 100: return None
        by_color = defaultdict(list)
        for r, c in non_bg: by_color[int(grid[int(r), int(c)])].append((int(r), int(c)))
        for color, coords in by_color.items():
            if len(coords) < 2: continue
            coords.sort()
            for i in range(len(coords)-1):
                r1, c1 = coords[i]; r2, c2 = coords[i+1]
                dr, dc = r2-r1, c2-c1
                if abs(dr) > 3 or abs(dc) > 3: continue
                nr, nc = r2+dr, c2+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if int(grid[nr, nc]) == self.bg_color and (nr, nc) not in self.interactions: return ((nr, nc), color)
        return None

    def _detect_symmetry_completion(self, grid, rows, cols):
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mc = cols - 1 - c
                val_l = int(grid[r, c]); val_r = int(grid[r, mc])
                if val_l != self.bg_color and val_r == self.bg_color:
                    if (r, mc) not in self.interactions: return ((r, mc), val_l)
                if val_r != self.bg_color and val_l == self.bg_color:
                     if (r, c) not in self.interactions: return ((r, c), val_r)
        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 50) & (unique != self.bg_color)]
        if rare.size == 0: return []
        matches = np.argwhere(np.isin(grid, rare))
        targets = []
        for r, c in matches:
            if (r, c) != start_rc:
                targets.append((abs(r-start_rc[0]) + abs(c-start_rc[1]), (r, c)))
        targets.sort()
        return targets

    def _scan_adjacent(self, rc, rows, cols):
        r, c = int(rc[0]), int(rc[1])
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols: out.append((nr, nc))
        return out

    def _find_frontier(self, start, rows, cols):
        if not self.control_map: return None
        q = deque([(start, [])]); seen = {start}; steps = 0
        while q:
            steps += 1;
            if steps > 200: break
            curr, path = q.popleft()
            if len(path) > 8: return path
            r, c = curr
            for act, (dr, dc) in self.control_map.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols): continue
                if (nr, nc) in self.walls: continue
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append(((nr, nc), path+[act]))
        return None
