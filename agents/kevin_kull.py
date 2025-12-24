import numpy as np
import random
import heapq
import time
from collections import deque, Counter, defaultdict
from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- OPTIONAL VISION STACK (Hardware Agnostic) ---
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM
    import timm
    import einops
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# --- SEMANTIC BRIDGE ---
COLOR_MAP = {
    "black": 0, "empty": 0, "void": 0, "background": 0,
    "blue": 1, "cyan": 1, "azure": 1,
    "red": 2, "crimson": 2, "maroon": 9, "burgundy": 9,
    "green": 3, "lime": 3, "forest": 3,
    "yellow": 4, "gold": 4,
    "grey": 5, "gray": 5, "silver": 5,
    "pink": 6, "magenta": 6, "purple": 6,
    "orange": 7, "brown": 7,
    "teal": 8, "turquoise": 8
}
ARC_COLORS = [(0,0,0), (0,116,217), (255,65,54), (46,204,64), (255,220,0),
              (170,170,170), (240,18,190), (255,133,27), (127,219,255), (135,12,37)]

# --- LOCAL VISION NEXUS ---
class LocalVisionNexus:
    _instance = None
    def __init__(self):
        self._model = None
        if not VISION_AVAILABLE: return
        try:
            # Hardware Agnostic Loading (CUDA -> MPS -> CPU)
            self._device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            dtype = torch.float16 if self._device != "cpu" else torch.float32

            # Load Florence-2-Base (Fast, Apache 2.0)
            model_id = "microsoft/Florence-2-base"
            self._model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(self._device).eval()
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except: pass

    @classmethod
    def get_instance(cls):
        if cls._instance is None: cls._instance = LocalVisionNexus()
        return cls._instance

    def analyze(self, grid_np):
        if not self._model: return [], []
        tags, targets = [], []
        try:
            h, w = grid_np.shape
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(10): img_rgb[grid_np == i] = ARC_COLORS[i]
            scale = max(1, 512 // max(h, w))
            pil_img = Image.fromarray(img_rgb).resize((w*scale, h*scale), Image.NEAREST)

            with torch.inference_mode():
                # Pass 1: Caption
                inputs = self._processor(text="<MORE_DETAILED_CAPTION>", images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                if self._device == "mps" and inputs["pixel_values"].dtype != torch.float32:
                     inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

                gen_ids = self._model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=64, num_beams=3)
                text = self._processor.batch_decode(gen_ids, skip_special_tokens=False)[0]

                if any(x in text.lower() for x in ["maze", "path", "wall"]): tags.append("NAV")
                if any(x in text.lower() for x in ["pattern", "grid", "symmetric"]): tags.append("PATTERN")

                # Pass 2: Object Detection
                inputs_od = self._processor(text="<OD>", images=pil_img, return_tensors="pt")
                inputs_od = {k: v.to(self._device) for k, v in inputs_od.items()}
                if self._device == "mps": inputs_od["pixel_values"] = inputs_od["pixel_values"].to(torch.float32)

                gen_od = self._model.generate(input_ids=inputs_od["input_ids"], pixel_values=inputs_od["pixel_values"], max_new_tokens=64, num_beams=3)
                text_od = self._processor.batch_decode(gen_od, skip_special_tokens=False)[0]
                res = self._processor.post_process_generation(text_od, task="<OD>", image_size=(pil_img.width, pil_img.height))["<OD>"]

                for bbox, label in zip(res.get('bboxes', []), res.get('labels', [])):
                    cy = int(((bbox[1]+bbox[3])/2)/scale); cx = int(((bbox[0]+bbox[2])/2)/scale)
                    if 0 <= cy < h and 0 <= cx < w: targets.append(((cy, cx), label))
        except: pass
        return tags, targets

# --- GLOBAL NEXUS ---
class GlobalNexus:
    CONTROLS = {}
    PERSPECTIVE_WEIGHTS = defaultdict(lambda: 1.0)
    USE_SCORES = defaultdict(lambda: 1.0)
    CLICK_SCORES = defaultdict(lambda: 1.0)
    VISION_CACHE = {}
    VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v11.0-EVENT_HORIZON
    1. AXIOM OF WALKABILITY: Removed 'val==0' check. Background is walkable.
    2. INTERACTION COOLDOWN: Dict-based memory allows re-clicking buttons.
    3. BACKGROUND INFERENCE: Dynamically detects floor color for A* costs.
    4. OBJECT PRESERVATION: Rare objects are not walls; they are interactions.
    """
    MAX_ACTIONS = 400
    VISION_INTERVAL = 15

    RAW_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
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
        self.interactions = {} # {coord: last_step_clicked} - COOLDOWN MAP
        self.mode = "HANDSHAKE"
        self.handshake_queue = deque(self.RAW_ACTIONS)
        self.last_pos = None
        self.last_grid = None
        self.last_action = None
        self.last_target_val = None
        self.agent_color = None
        self.bg_color = 0 # Dynamically inferred

        self.stuck_counter = 0
        self.lost_counter = 0
        self.step_count = 0
        self.action_data = {}
        self.vision_engine = LocalVisionNexus.get_instance() if VISION_AVAILABLE else None
        self.vision_targets = []

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state in [GameState.WIN, GameState.GAME_OVER]

    def choose_action(self, frames: list[FrameData], latest: FrameData) -> GameAction:
        self.action_data = {}
        self.step_count += 1
        if not self.control_map and GlobalNexus.CONTROLS:
            self.control_map = GlobalNexus.CONTROLS.copy()
            self.mode = "AVATAR"
        if latest.state == GameState.NOT_PLAYED or not latest.frame: return GameAction.RESET
        try: return self._horizon_loop(latest)
        except: return random.choice(self.RAW_ACTIONS)

    def _horizon_loop(self, latest: FrameData):
        grid = self._parse_grid(latest)
        rows, cols = grid.shape
        current_hash = grid.tobytes()

        # 0. INFER BACKGROUND (Fix: Floor is Lava)
        vals, counts = np.unique(grid, return_counts=True)
        if len(vals) > 0:
            self.bg_color = vals[np.argmax(counts)]

        # 1. VISION (Event Driven)
        inv_changed = (np.sum(grid) != (np.sum(self.last_grid) if self.last_grid is not None else 0))
        if self.vision_engine and (self.step_count % self.VISION_INTERVAL == 0 or inv_changed):
            if current_hash not in GlobalNexus.VISION_CACHE:
                tags, targets = self.vision_engine.analyze(grid)
                GlobalNexus.VISION_CACHE[current_hash] = (tags, targets)
            tags, targets = GlobalNexus.VISION_CACHE[current_hash]
            self.vision_targets = targets

            GlobalNexus.VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}
            if "NAV" in tags: GlobalNexus.VISION_HINTS["Newton"] = 1.5
            if "PATTERN" in tags: GlobalNexus.VISION_HINTS["Euclid"] = 1.5
            if "INTERACT" in tags: GlobalNexus.VISION_HINTS["Skinner"] = 1.5

        # 2. DELTA IDENTITY
        agent_rc = self._locate_agent(latest, grid)
        if agent_rc is None and self.last_grid is not None:
             if grid.shape == self.last_grid.shape:
                 delta_rc = self._find_center_of_change(self.last_grid, grid)
                 if delta_rc:
                     agent_rc = delta_rc
                     self.agent_color = grid[agent_rc]
        self.lost_counter = self.lost_counter + 1 if agent_rc is None else 0

        # 3. PHYSICS
        self._update_physics(grid, agent_rc, np.sum(grid))

        if self.mode == "HANDSHAKE": return self._run_handshake(grid, agent_rc)

        # 4. PARLIAMENT
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

        # 5. EXECUTIVE
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids: return self._perspective_chaos()

        weighted_bids = []
        for score, action, target, owner in valid_bids:
            w_score = score * GlobalNexus.PERSPECTIVE_WEIGHTS[owner]
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            weighted_bids.append((w_score, action, target, owner))

        weighted_bids.sort(key=lambda x: x[0], reverse=True)
        score, action, target, owner = weighted_bids[0]

        # 6. EXECUTION
        if target:
            val = 1
            if len(target) > 2: val = target[2]
            else:
                try:
                    gval = grid[target[0], target[1]]
                    val = gval if gval != 0 else 1
                except: pass
            # Force color if painting empty space (unless Euclid specified)
            if val == 0 and owner != "Euclid": val = 1

            self._set_payload(target[:2], val)
            self.last_target_val = val
            if action in [self.ACT_CLICK, self.ACT_USE]:
                self.interactions[target[:2]] = self.step_count # Record Timestamp
        else:
            self.last_target_val = None

        self.last_action = action
        return action

    # --- PERSPECTIVES ---
    def _perspective_oracle(self, grid):
        for (r,c), label in self.vision_targets:
            if (self.step_count - self.interactions.get((r,c), -100) > 20):
                val = 1
                for name, v_int in COLOR_MAP.items():
                    if name in label.lower(): val = v_int; break
                return (0.97, self.ACT_CLICK, (r,c,val), f"Oracle({label})")
        return (0.0, None, None, "Oracle")

    def _perspective_newton(self, grid, agent_rc, rows, cols, boost=1.0):
        if not agent_rc: return (0.0, None, None, "Newton")
        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue
            path = self._astar(grid, agent_rc, t_rc, rows, cols, allow_push=True)
            if path: return (0.95 * boost, path[0], t_rc, "Newton")
            else: self.bad_goals.add(t_rc)
        frontier = self._find_frontier(grid, agent_rc, rows, cols)
        if frontier: return (0.6 * boost, frontier[0], None, "Newton")
        return (0.0, None, None, "Newton")

    def _perspective_euclid(self, grid, rows, cols, boost=1.0):
        # Vector Extension
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            target, color = ext
            return (0.92 * boost, self.ACT_CLICK, (target[0], target[1], color), "Euclid")
        # Symmetry
        sym = self._detect_symmetry_completion(grid, rows, cols)
        if sym:
            target, color = sym
            return (0.90 * boost, self.ACT_CLICK, (target[0], target[1], color), "EuclidSym")
        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols, boost=1.0):
        if agent_rc:
            adj = self._scan_adjacent(grid, agent_rc, rows, cols)
            for r, c in adj:
                val = grid[r, c]
                # Interact if not Background and not recently clicked
                if val != self.bg_color and (self.step_count - self.interactions.get((r,c), -100) > 10):
                    score = 0.8 * GlobalNexus.USE_SCORES[val] * boost
                    return (score, self.ACT_USE, (r,c), "Skinner")
        if self.mode != "AVATAR" or self.lost_counter > 20:
            unique, counts = np.unique(grid, return_counts=True)
            for val in unique[np.argsort(counts)]:
                if val == self.bg_color: continue
                matches = np.argwhere(grid == val)
                for r, c in matches:
                    if (self.step_count - self.interactions.get((r,c), -100) > 20):
                         score = 0.6 * GlobalNexus.CLICK_SCORES[val] * boost
                         return (score, self.ACT_CLICK, (r,c, val), "Skinner")
        return (0.0, None, None, "Skinner")

    def _perspective_fusion(self, grid, agent_rc, rows, cols):
        if not agent_rc: return (0.0, None, None, "Fusion")
        targets = self._scan_targets(grid, agent_rc)
        for dist, t_rc in targets:
            # Only fuse if target is active (not cooling down)
            if dist > 1 and (self.step_count - self.interactions.get(t_rc, -100) > 20):
                val = grid[t_rc]
                adj = self._scan_adjacent(grid, t_rc, rows, cols)
                for near_rc in adj:
                    if grid[near_rc] == self.bg_color: # Use BG color
                        path = self._astar(grid, agent_rc, near_rc, rows, cols)
                        if path:
                            relevance = max(GlobalNexus.USE_SCORES[val], GlobalNexus.CLICK_SCORES[val])
                            return (0.94 * relevance, path[0], t_rc, "Fusion")
        return (0.0, None, None, "Fusion")

    def _perspective_recovery(self):
        if self.stuck_counter > 2:
            if self.stuck_counter > 5: return (3.0, self._perspective_chaos(), None, "Recovery")
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
        if self.handshake_queue: return self.handshake_queue.popleft()
        else:
            self.mode = "AVATAR"
            if not self.control_map: self.control_map = self.DEFAULT_CONTROLS.copy()
            else:
                for k, v in self.DEFAULT_CONTROLS.items():
                    if k not in self.control_map: self.control_map[k] = v
            GlobalNexus.CONTROLS = self.control_map.copy()
            return random.choice(self.RAW_ACTIONS)

    def _update_physics(self, grid, agent_rc, current_inv):
        prev_hash = self.last_grid.tobytes() if self.last_grid is not None else b""
        if self.last_action in [self.ACT_CLICK, self.ACT_USE, self.ACT_INTERACT] and self.last_grid is not None:
             success = (current_inv != np.sum(self.last_grid))
             val = self.last_target_val
             if val is not None:
                 dic = GlobalNexus.USE_SCORES if self.last_action == self.ACT_USE else GlobalNexus.CLICK_SCORES
                 dic[val] *= (1.2 if success else 0.9)

        moved = (agent_rc != self.last_pos) if self.last_pos and agent_rc else False
        changed = (current_inv != np.sum(self.last_grid)) if self.last_grid is not None else True
        if not changed and not moved: self.stuck_counter += 1
        else: self.stuck_counter = 0

        # SMART WALL MARKING (Object Preservation)
        if self.mode == "AVATAR" and self.last_pos and agent_rc == self.last_pos:
             if self.last_action in self.control_map:
                 dr, dc = self.control_map[self.last_action]
                 tr, tc = self.last_pos[0]+dr, self.last_pos[1]+dc

                 # Only mark if Common (BG/Wall). If Rare (Goal/Block), force interaction.
                 val = grid[tr, tc] if 0 <= tr < grid.shape[0] and 0 <= tc < grid.shape[1] else 0
                 unique, counts = np.unique(grid, return_counts=True)
                 count = counts[np.where(unique==val)][0] if val in unique else 999

                 if count > 50 or val == self.bg_color:
                     self.walls.add((tr, tc))
                 else:
                     # RARE OBJECT COLLISION -> FORCE INTERACTION
                     self.interactions[((tr, tc))] = -100 # Expire cooldown
                     self.stuck_counter = 3 # Trigger Recovery

                 if self.stuck_counter > 8: self.agent_color = None

        self.last_pos = agent_rc
        self.last_grid = grid.copy()

    # --- HELPERS ---
    def _parse_grid(self, latest):
        raw = np.array(latest.frame)
        if raw.ndim == 2: return raw
        if raw.ndim == 3: return raw[-1]
        return raw.reshape((int(np.sqrt(raw.size)), -1))

    def _astar(self, grid, start, end, rows, cols, allow_push=False):
        if not self.control_map: return None
        pq = [(0, 0, start, [])]; best_g = {start: 0}
        while pq:
            _, steps, curr, path = heapq.heappop(pq)
            if curr == end: return path
            if steps > 300: break
            r, c = curr
            for act, (dr, dc) in self.control_map.items():
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in self.walls: continue
                    val = grid[nr, nc]
                    # Walkable if BG or Goal
                    is_walk = (val == self.bg_color) or ((nr, nc) == end)
                    if allow_push: is_walk = True
                    if is_walk:
                        cost = 1 if val == self.bg_color else 5
                        ng = steps + cost
                        if (nr, nc) not in best_g or ng < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = ng
                            heapq.heappush(pq, (ng + abs(nr-end[0]) + abs(nc-end[1]), ng, (nr, nc), path+[act]))
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
            if c == self.bg_color: continue
            coords = np.argwhere(grid == c)
            if len(coords) > 0: return tuple(coords[0])
        return None

    def _find_center_of_change(self, prev, curr):
        if prev.shape != curr.shape: return None
        diff = prev != curr
        coords = np.argwhere(diff)
        if len(coords) == 0: return None
        return (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))

    def _find_vector_extension(self, grid, rows, cols):
        non_zeros = np.argwhere(grid != self.bg_color)
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
                    if grid[nr, nc] == self.bg_color: # Valid on BG
                         if (nr, nc) not in self.interactions: return ((nr, nc), color)
        return None

    def _detect_symmetry_completion(self, grid, rows, cols):
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mc = cols - 1 - c
                val_l = grid[r, c]; val_r = grid[r, mc]
                if val_l != self.bg_color and val_r == self.bg_color:
                    if (r, mc) not in self.interactions: return ((r, mc), val_l)
                if val_r != self.bg_color and val_l == self.bg_color:
                     if (r, c) not in self.interactions: return ((r, c), val_r)
        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 50) & (unique != self.bg_color)]
        targets = []
        matches = np.argwhere(np.isin(grid, rare))
        for r, c in matches:
            if (r, c) != start_rc:
                targets.append((abs(r-start_rc[0]) + abs(c-start_rc[1]), (r, c)))
        targets.sort()
        return targets

    def _scan_adjacent(self, grid, rc, rows, cols):
        r, c = rc; adj = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols: adj.append((nr, nc))
        return adj

    def _set_payload(self, rc, val):
        r, c = rc
        # Correct transposition for API
        payload = {'x': int(c), 'y': int(r), 'value': int(val)}
        self.action_data = payload
        try: self.ACT_CLICK.action_data = payload
        except: pass

    def _find_frontier(self, grid, start, rows, cols):
        if not self.control_map: return None
        q = deque([(start, [])]); seen = {start}; steps = 0
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
