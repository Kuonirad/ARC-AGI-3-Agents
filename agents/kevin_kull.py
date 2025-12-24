import numpy as np
import random
import heapq
import time
from collections import deque, Counter, defaultdict
from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- OPTIONAL VISION STACK (Lazy Import) ---
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# --- ARC COLOR PALETTE (RGB) ---
ARC_COLORS = [
    (0, 0, 0),       # 0: Black
    (0, 116, 217),   # 1: Blue
    (255, 65, 54),   # 2: Red
    (46, 204, 64),   # 3: Green
    (255, 220, 0),   # 4: Yellow
    (170, 170, 170), # 5: Grey
    (240, 18, 190),  # 6: Pink
    (255, 133, 27),  # 7: Orange
    (127, 219, 255), # 8: Teal
    (135, 12, 37)    # 9: Maroon
]

# --- LOCAL VISION NEXUS (Singleton) ---
class LocalVisionNexus:
    """
    Local VLM Driver (Florence-2-Base).
    Singleton pattern prevents reloading heavy weights per episode.
    """
    _instance = None
    _model = None
    _processor = None
    _device = "cpu"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LocalVisionNexus()
        return cls._instance

    def __init__(self):
        if not VISION_AVAILABLE: return
        try:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            # Florence-2-base: Lightweight (0.23B), Apache 2.0, Fast
            model_id = "microsoft/Florence-2-base"

            print(f"[NOVA-VISION] Loading {model_id} on {self._device}...")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            ).to(self._device).eval()
            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            print("[NOVA-VISION] Stack Online.")
        except Exception as e:
            print(f"[NOVA-VISION] Init Failed (Running Symbolic): {e}")
            self._model = None

    def analyze(self, grid_np):
        """
        Returns: (tags: list[str], targets: list[((r,c), label)])
        """
        if not self._model: return [], []

        tags = []
        targets = []

        try:
            # 1. Render Grid to RGB Image
            h, w = grid_np.shape
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(10):
                img_rgb[grid_np == i] = ARC_COLORS[i]

            # Upscale for VLM visibility (min ~300px)
            scale = max(1, 512 // max(h, w))
            pil_img = Image.fromarray(img_rgb).resize((w*scale, h*scale), Image.NEAREST)

            # 2. PASS 1: Captioning (Context)
            task_cap = "<MORE_DETAILED_CAPTION>"
            inputs = self._processor(text=task_cap, images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            gen_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=64,
                do_sample=False,
                num_beams=3
            )
            text_cap = self._processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
            res_cap = self._processor.post_process_generation(
                text_cap, task=task_cap, image_size=(pil_img.width, pil_img.height)
            )[task_cap].lower()

            # Extract Semantic Tags
            if any(w in res_cap for w in ["maze", "path", "room", "corridor", "wall"]): tags.append("NAV")
            if any(w in res_cap for w in ["pattern", "grid", "line", "symmetry", "mosaic"]): tags.append("PATTERN")
            if any(w in res_cap for w in ["object", "shape", "button", "switch", "separate"]): tags.append("INTERACT")

            # 3. PASS 2: Object Detection (Targets)
            task_od = "<OD>"
            inputs_od = self._processor(text=task_od, images=pil_img, return_tensors="pt")
            inputs_od = {k: v.to(self._device) for k, v in inputs_od.items()}

            gen_ids_od = self._model.generate(
                input_ids=inputs_od["input_ids"],
                pixel_values=inputs_od["pixel_values"],
                max_new_tokens=64,
                do_sample=False,
                num_beams=3
            )
            text_od = self._processor.batch_decode(gen_ids_od, skip_special_tokens=False)[0]
            res_od = self._processor.post_process_generation(
                text_od, task=task_od, image_size=(pil_img.width, pil_img.height)
            )[task_od]

            # Map BBoxes back to Grid
            for bbox, label in zip(res_od.get('bboxes', []), res_od.get('labels', [])):
                # bbox: [x1, y1, x2, y2]
                cy = int(((bbox[1] + bbox[3]) / 2) / scale)
                cx = int(((bbox[0] + bbox[2]) / 2) / scale)
                if 0 <= cy < h and 0 <= cx < w:
                    targets.append(((cy, cx), label))

        except Exception as e:
            # print(f"[NOVA-VISION] Inference Error: {e}")
            pass

        return tags, targets

# --- GLOBAL MEMORY ---
class GlobalNexus:
    CONTROLS = {}
    PERSPECTIVE_WEIGHTS = defaultdict(lambda: 1.0)
    USE_SCORES = defaultdict(lambda: 1.0)
    CLICK_SCORES = defaultdict(lambda: 1.0)
    # Shared Vision Cache
    VISION_CACHE = {} # hash -> (tags, targets)
    VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v10.0-COSMOS-FREE (KEVIN_KULL)
    The 'Open-Perceptual' Agent.

    STACK:
    1. VISION NEXUS: Local Florence-2 VLM for captioning & object detection.
    2. SEMANTIC MODULATION: Captions boost relevant perspectives.
    3. VISUAL TARGETING: OD Bounding boxes become high-confidence click targets.
    4. COSMOS CORE (v9.1): Robust physics, loop breaking, and delta identity.
    """

    MAX_ACTIONS = 400
    VISION_INTERVAL = 15

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
        self.step_count = 0
        self.action_data = {}

        # Vision State
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
        current_hash = grid.tobytes()

        # 2. VISION ORACLE (Throttled & Cached)
        if self.vision_engine and (self.step_count % self.VISION_INTERVAL == 0 or self.stuck_counter > 5):
            if current_hash not in GlobalNexus.VISION_CACHE:
                tags, targets = self.vision_engine.analyze(grid)
                GlobalNexus.VISION_CACHE[current_hash] = (tags, targets)

            tags, targets = GlobalNexus.VISION_CACHE[current_hash]
            self.vision_targets = targets

            # Semantic Modulation
            GlobalNexus.VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}
            if "NAV" in tags: GlobalNexus.VISION_HINTS["Newton"] = 1.5
            if "PATTERN" in tags: GlobalNexus.VISION_HINTS["Euclid"] = 1.5
            if "INTERACT" in tags: GlobalNexus.VISION_HINTS["Skinner"] = 1.5

        # 3. DELTA IDENTITY
        agent_rc = self._locate_agent(latest, grid)
        if agent_rc is None and self.last_grid is not None:
             if grid.shape == self.last_grid.shape:
                 delta_rc = self._find_center_of_change(self.last_grid, grid)
                 if delta_rc:
                     agent_rc = delta_rc
                     self.agent_color = grid[agent_rc]

        if agent_rc is None: self.lost_counter += 1
        else: self.lost_counter = 0

        # 4. PHYSICS UPDATE
        self._update_physics(grid, agent_rc, np.sum(grid))

        # 5. HANDSHAKE
        if self.mode == "HANDSHAKE":
            return self._run_handshake(grid, agent_rc)

        # 6. PARLIAMENT (Boosted)
        bids = []
        nav_b = GlobalNexus.VISION_HINTS["Newton"]
        pat_b = GlobalNexus.VISION_HINTS["Euclid"]
        int_b = GlobalNexus.VISION_HINTS["Skinner"]

        # [Blind Pilot]
        if self.mode == "AVATAR" and agent_rc is None:
            bid = 0.99 if self.lost_counter < 40 else 0.0
            bids.append((bid, self._perspective_chaos(), None, "BlindPilot"))

        # [Oracle] (Visual Targets)
        bids.append(self._perspective_oracle(grid))

        # [Newton]
        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols, nav_b))
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        # [Euclid]
        bids.append(self._perspective_euclid(grid, rows, cols, pat_b))

        # [Skinner]
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols, int_b))

        # [Recovery]
        bids.append(self._perspective_recovery())

        # 7. EXECUTIVE
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids: return self._perspective_chaos()

        weighted_bids = []
        for score, action, target, owner in valid_bids:
            w_score = score * GlobalNexus.PERSPECTIVE_WEIGHTS[owner]
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            weighted_bids.append((w_score, action, target, owner))

        weighted_bids.sort(key=lambda x: x[0], reverse=True)
        best_bid = weighted_bids[0]
        score, action, target, owner = best_bid

        # 8. EXECUTION
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
    def _perspective_oracle(self, grid):
        # Bids on visually detected objects
        for (r,c), label in self.vision_targets:
            if (r,c) not in self.interactions:
                # High base confidence for VLM detection
                return (0.97, self.ACT_CLICK, (r,c,1), f"Oracle({label})")
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
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            target, color = ext
            return (0.92 * boost, self.ACT_CLICK, (target[0], target[1], color), "Euclid")
        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols, boost=1.0):
        if agent_rc:
            adj = self._scan_adjacent(grid, agent_rc, rows, cols)
            for r, c in adj:
                val = grid[r, c]
                if val != 0 and (r,c) not in self.interactions:
                    score = 0.8 * GlobalNexus.USE_SCORES[val] * boost
                    return (score, self.ACT_USE, (r,c), "Skinner")
        if self.mode != "AVATAR" or self.lost_counter > 20:
            unique, counts = np.unique(grid, return_counts=True)
            for val in unique[np.argsort(counts)]:
                if val == 0: continue
                matches = np.argwhere(grid == val)
                for r, c in matches:
                    if (r, c) not in self.interactions:
                         score = 0.6 * GlobalNexus.CLICK_SCORES[val] * boost
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
        if self.stuck_counter > 2:
            if self.stuck_counter > 5:
                return (3.0, self._perspective_chaos(), None, "Recovery")
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

    def _update_physics(self, grid, agent_rc, current_hash):
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

        state_changed = (current_hash != np.sum(self.last_grid)) if self.last_grid is not None else True
        moved = (agent_rc != self.last_pos) if self.last_pos is not None and agent_rc is not None else False

        if not state_changed and not moved: self.stuck_counter += 1
        else: self.stuck_counter = 0

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
        return (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))

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
