import numpy as np
import random
import heapq
import time
from collections import deque, Counter, defaultdict
from .agent import Agent
from .structs import FrameData, GameAction, GameState

# --- OPTIONAL VISION STACK (Lazy Import & Hardware Agnostic) ---
# --- OPTIONAL VISION STACK (Lazy Import) ---
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForCausalLM
    # Check for dependencies to fail fast safely
    import timm
    import einops
    VISION_AVAILABLE = True
except ImportError:
    print("[NOVA] Vision dependencies missing. Running Symbolic.")
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

# --- SEMANTIC BRIDGE ---
COLOR_MAP = {
    "black": 0, "empty": 0, "void": 0,
    "blue": 1, "cyan": 1, "azure": 1,
    "red": 2, "crimson": 2, "scarlet": 2,
    "green": 3, "lime": 3, "forest": 3,
    "yellow": 4, "gold": 4, "lemon": 4,
    "grey": 5, "gray": 5, "silver": 5,
    "pink": 6, "magenta": 6, "purple": 6, "rose": 6,
    "orange": 7, "brown": 7, "amber": 7,
    "teal": 8, "turquoise": 8, "cyan": 8,
    "maroon": 9, "burgundy": 9, "dark red": 9
}

# --- LOCAL VISION NEXUS (Singleton) ---
class LocalVisionNexus:
    """
    Local VLM Driver (Florence-2-Base).
    Singleton with Hardware Agnosticism (CUDA/MPS/CPU) & Inference Mode.
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
            # Hardware Selection: Critical for Performance
            if torch.cuda.is_available():
                self._device = "cuda"
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                self._device = "mps"  # Apple Silicon Support
                dtype = torch.float16
            else:
                self._device = "cpu"
                dtype = torch.float32

            model_id = "microsoft/Florence-2-base"
            # print(f"[NOVA-VISION] Loading {model_id} on {self._device}...")

            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype
            ).to(self._device).eval()

            self._processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception as e:
            print(f"[NOVA-VISION] Init Failed: {e}")
            self._model = None

    def analyze(self, grid_np):
        if not self._model: return [], []
        tags = []
        targets = []
        try:
            h, w = grid_np.shape
            img_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(10): img_rgb[grid_np == i] = ARC_COLORS[i]

            scale = max(1, 512 // max(h, w))
            pil_img = Image.fromarray(img_rgb).resize((w*scale, h*scale), Image.NEAREST)

            # OPTIMIZATION: Torch Inference Mode (No Grad)
            with torch.inference_mode():
                # Pass 1: Caption
                task_cap = "<MORE_DETAILED_CAPTION>"
                inputs = self._processor(text=task_cap, images=pil_img, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                # Fix for MPS dtype mismatch
                if self._device == "mps" and inputs["pixel_values"].dtype != torch.float32:
                     inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)

                gen_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=64,
                    num_beams=3
                )
                text_cap = self._processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
                res_cap = self._processor.post_process_generation(
                    text_cap, task=task_cap, image_size=(pil_img.width, pil_img.height)
                )[task_cap].lower()

                if any(w in res_cap for w in ["maze", "path", "corridor"]): tags.append("NAV")
                if any(w in res_cap for w in ["pattern", "symmetry", "mirror"]): tags.append("PATTERN")
                if any(w in res_cap for w in ["object", "button", "switch"]): tags.append("INTERACT")

                # Pass 2: Detection
                task_od = "<OD>"
                inputs_od = self._processor(text=task_od, images=pil_img, return_tensors="pt")
                inputs_od = {k: v.to(self._device) for k, v in inputs_od.items()}
                if self._device == "mps":
                    inputs_od["pixel_values"] = inputs_od["pixel_values"].to(torch.float32)

                gen_ids_od = self._model.generate(
                    input_ids=inputs_od["input_ids"],
                    pixel_values=inputs_od["pixel_values"],
                    max_new_tokens=64,
                    num_beams=3
                )
                text_od = self._processor.batch_decode(gen_ids_od, skip_special_tokens=False)[0]
                res_od = self._processor.post_process_generation(
                    text_od, task=task_od, image_size=(pil_img.width, pil_img.height)
                )[task_od]

                for bbox, label in zip(res_od.get('bboxes', []), res_od.get('labels', [])):
                    cy = int(((bbox[1] + bbox[3]) / 2) / scale)
                    cx = int(((bbox[0] + bbox[2]) / 2) / scale)
                    # SAFETY CLAMP: Prevent Hallucinated Coordinates
                    cy = max(0, min(cy, h - 1))
                    cx = max(0, min(cx, w - 1))
                    targets.append(((cy, cx), label))
        except Exception: pass
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
    VISION_CACHE = {}
    VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}

    @classmethod
    def clean_cache(cls):
        # Hygiene: Prevent Memory Leaks
        if len(cls.VISION_CACHE) > 100: cls.VISION_CACHE.clear()

class KevinKullAgent(Agent):
    MAX_ACTIONS = 400
    VISION_INTERVAL = 15

    RAW_ACTIONS = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]
    DEFAULT_CONTROLS = {
        GameAction.ACTION1: (-1, 0), GameAction.ACTION2: (1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)
    }
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

    The 'Interaction Loophole' Patch.

    PATCH LOG:
    1. STUCK DETECTION: Now couples Position Delta + Grid Hash Delta.
       Prevents infinite ACTION5/6 loops.
    2. RECOVERY: Explicitly breaks loops via forced entropy or alternative interaction.
    3. MAX ACTIONS: Lifted to 400 to allow recovery from deep stalls.
    4. DELTA IDENTITY: Locates agent via grid change if color tracking fails.
    """

    MAX_ACTIONS = 400 # Requesting extended runtime

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
    DEFAULT_CONTROLS = {
        GameAction.ACTION1: (-1, 0), GameAction.ACTION2: (1, 0),
        GameAction.ACTION3: (0, -1), GameAction.ACTION4: (0, 1)
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
        # --- Local Memory (Per Level) ---
        self.control_map = {}     # Synced from Nexus
        self.walls = set()
        self.bad_goals = set()
        self.interactions = set()

        # State
        # --- State ---
        self.mode = "HANDSHAKE"
        self.handshake_queue = deque(self.RAW_ACTIONS)
        self.last_pos = None
        self.last_grid = None
        self.last_action = None
        self.last_target_val = None
        self.agent_color = None
        self.inventory_hash = 0

        self.stuck_counter = 0
        self.lost_counter = 0
        self.step_count = 0
        self.action_data = {}

        self.vision_engine = LocalVisionNexus.get_instance() if VISION_AVAILABLE else None
        self.vision_targets = []
        # Vision State
        self.vision_engine = LocalVisionNexus.get_instance() if VISION_AVAILABLE else None
        self.vision_targets = []
        self.stuck_counter = 0
        self.lost_counter = 0
        self.last_perspective = None # Track for diagnostics
        self.agent_color = None
        self.stuck_counter = 0
        self.lost_counter = 0
        self.inventory_hash = 0
        self.last_perspective = None
        self.agent_color = None
        self.stuck_counter = 0
        self.inventory_hash = 0
from collections import deque, Counter
from .agent import Agent
from .structs import FrameData, GameAction, GameState

class KevinKullAgent(Agent):
    """
    NOVA-M-COP v4.0-OMEGA (KEVIN_KULL)
    The 'Recursive Synthesis' Agent.

    HIERARCHY:
    1. EXECUTIVE (Metacognition): Resolves conflicts via Recursive Critique (Reachability Check).
    2. PERSPECTIVES (The Parliament):
       - NEWTON (Nav): Soft A* with Dynamic Obstacle Costing.
       - EUCLID (Pattern): Vector Extrapolation + Symmetry Mirroring.
       - SKINNER (Causal): Interaction weighted by History of Effect.
    3. SUBSTRATE (Physics):
       - Dynamic Identity (Shape-Shifting).
       - Autopoietic Calibration (Wiggle-Test).

    CRITICAL FIX v4.0: Sanitized Bidding (Prevents Payload Race Conditions).
    NOVA-M-COP v3.0-KEVIN_KULL
    The 'Optimistic' Polymath.

    PHILOSOPHY:
    - Optimistic Physics: Assume everything is walkable until we crash.
    - Wiggle-Calibration: Try all 4 directions before giving up on having a body.
    - Dynamic Identity: Tracking logic that handles color/shape shifts (Keys, Portals).
class NovaSingularityAgent(Agent):
    """
    NOVA-M-COP v2.0-SINGULARITY
    The 'Polymathic' Generalist.

    CAPABILITIES:
    1. AVATAR: Solves Navigation (ls20) via Dynamic A* & State-Dependent Physics.
    2. SCIENTIST: Solves Orchestration (vc33) via Rarity probing.
    3. BUILDER: Solves Patterning (ft09) via Vector Extrapolation (Line Extension).

    FIXES:
    - Coordinate Hygiene: Strict (row, col) management.
    - Action Payload: Robustly attaches (x,y) data to clicks via multiple shims.
    """

    # Robust Action Map
    ACT_UP = GameAction.ACTION1
    ACT_DOWN = GameAction.ACTION2
    ACT_LEFT = GameAction.ACTION3
    ACT_RIGHT = GameAction.ACTION4
    ACT_USE = getattr(GameAction, 'ACTION5', GameAction.ACTION5)   # Space
    ACT_CLICK = getattr(GameAction, 'ACTION6', GameAction.ACTION6) # Click

    MAX_ACTIONS = 400

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # --- Knowledge Graph ---
        self.walls = set()        # Hard constraints
        self.bad_goals = set()    # Unreachable
        self.causal_history = defaultdict(float) # ObjectVal -> ImpactScore
        self.interactions = set() # Locations clicked

        # --- State ---
        self.mode = "CALIBRATING"
        self.boot_steps = 0
        self.last_state = None    # (rc, hash)
        self.last_grid = None
        self.last_action = None
        self.last_pos = None
        self.stuck_counter = 0
        self.inventory_hash = 0
        self.agent_color = None
        self.action_data = {}

        # --- Calibration ---
        self.calibration_moves = [
            self.ACT_RIGHT, self.ACT_DOWN, self.ACT_LEFT, self.ACT_UP
        ]
        self.walls = set()        # {(r,c)} - Proven Obstacles
        self.interactions = set() # {(r,c)} - Clicked locations
        self.bad_goals = set()    # {(r,c)} - Unreachable targets

        # --- State Tracking ---
        self.mode = "CALIBRATING"
        self.calibration_moves = [
            self.ACT_RIGHT, self.ACT_DOWN, self.ACT_LEFT, self.ACT_UP
        ]
        self.plan = deque()
        self.last_state = None    # (agent_rc, grid_hash, grid_bytes)
        self.last_grid = None     # Full grid
        self.last_action = None
        self.last_pos = None      # (r,c)
        self.stuck_counter = 0
        self.inventory_hash = 0
        self.agent_color = None   # Dynamic Identity
        self.boot_steps = 0
        self.walls = set()        # {(r,c)} - Obstacles
        self.interactions = set() # {(r,c)} - Clicked locations
        self.bad_goals = set()    # {(r,c)} - Unreachable/Useless targets

        # --- State Tracking ---
        self.mode = "CALIBRATING"
        self.plan = deque()
        self.last_state = None    # (agent_rc, grid_hash)
        self.last_action = None
        self.stuck_counter = 0
        self.inventory_hash = 0
        self.boot_steps = 0
        self.agent_color = None

        # --- Payload Shim ---
        self.action_data = {}

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
        try: return self._cosmos_loop(latest)
        except Exception: return random.choice(self.RAW_ACTIONS)

    def _cosmos_loop(self, latest: FrameData) -> GameAction:
        grid = self._parse_grid(latest)
        rows, cols = grid.shape
        # STRONG HASHING (Physics)
        current_hash = grid.tobytes()
        current_inv = np.sum(grid)

        # 1. VISION ORACLE (Event-Driven)
        # Run if: Timer OR Stuck OR Inventory Change
        inv_changed = (abs(current_inv - self.inventory_hash) > 0)
        should_see = (self.step_count % self.VISION_INTERVAL == 0) or \
                     (self.stuck_counter > 5) or inv_changed

        if self.vision_engine and should_see:
            GlobalNexus.clean_cache()

        # Fleet Sync
        # Sync with Fleet
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

            # Modulate
            # Semantic Modulation
            GlobalNexus.VISION_HINTS = {"Newton": 1.0, "Euclid": 1.0, "Skinner": 1.0}
            if "NAV" in tags: GlobalNexus.VISION_HINTS["Newton"] = 1.5
            if "PATTERN" in tags: GlobalNexus.VISION_HINTS["Euclid"] = 1.5
            if "INTERACT" in tags: GlobalNexus.VISION_HINTS["Skinner"] = 1.5

        # 2. DELTA IDENTITY
        agent_rc = self._locate_agent(latest, grid)
        if agent_rc is None and self.last_grid is not None:
        # 3. DELTA IDENTITY
        agent_rc = self._locate_agent(latest, grid)
        if agent_rc is None and self.last_grid is not None:
            return self._zenith_loop(latest)
        except Exception:
            return random.choice(self.RAW_ACTIONS)

    def _zenith_loop(self, latest: FrameData) -> GameAction:
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

        # 3. PHYSICS
        self._update_physics(grid, agent_rc, current_hash)

        if self.mode == "HANDSHAKE": return self._run_handshake(grid, agent_rc)

        # 4. PARLIAMENT

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

        # 3. PHYSICS UPDATE (The Fix: Progress-Decoupled)
        # 3. PHYSICS UPDATE (THE PATCH)
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

        # [Blind Pilot] (Priority 0.99 if lost)
        # [Blind Pilot]
        if self.mode == "AVATAR" and agent_rc is None:
            bid = 0.99 if self.lost_counter < 40 else 0.0
            bids.append((bid, self._perspective_chaos(), None, "BlindPilot"))

        bids.append(self._perspective_oracle(grid))

        # [Oracle] (Visual Targets)
        bids.append(self._perspective_oracle(grid))

        # [Newton]
        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols, nav_b))
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        bids.append(self._perspective_euclid(grid, rows, cols, pat_b))
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols, int_b))
        bids.append(self._perspective_recovery())

        # 5. EXECUTIVE
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids: return self._perspective_chaos()
        # [Euclid]
        bids.append(self._perspective_euclid(grid, rows, cols, pat_b))

        # [Skinner]
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols, int_b))

        # [Recovery]
        bids.append(self._perspective_recovery())

        # 7. EXECUTIVE
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids: return self._perspective_chaos()
        # [Newton] Nav
        # [Newton]
        # Newton (Nav)
        if self.mode == "AVATAR":
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols))
            bids.append(self._perspective_fusion(grid, agent_rc, rows, cols))

        # [Euclid] Pattern
        bids.append(self._perspective_euclid(grid, rows, cols))

        # [Skinner] Interact
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # [Recovery] (Triggers on Global Stasis)
        # [Euclid]
        bids.append(self._perspective_euclid(grid, rows, cols))

        # [Skinner]
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # [Recovery] (Now triggers correctly due to stuck fix)
        bids.append(self._perspective_recovery())

        # 6. L2: EXECUTIVE
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids: return self._perspective_chaos()
        # Euclid (Pattern) - Always active for vectors
        bids.append(self._perspective_euclid(grid, rows, cols))

        # Skinner (Interact) - GATED
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # 6. L2: EXECUTIVE
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
        self.action_data = {} # Clear payload shim
        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET
        try:
            return self._omega_executive(latest)
        except Exception:
            return self._perspective_chaos()

    def _omega_executive(self, latest: FrameData) -> GameAction:
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape
        current_hash = np.sum(grid)

        # 1. PERCEPTION & PHYSICS UPDATE
        agent_rc = self._locate_agent(latest, grid)
        self._update_physics(grid, agent_rc, current_hash)

        # 2. CALIBRATION PHASE
        if self.mode == "CALIBRATING":
            return self._run_calibration(grid, agent_rc)

        # 3. PERSPECTIVE BIDDING (Sanitized: No side effects yet)
        bids = []

        # Newton (Survival/Nav)
        if self.mode == "AVATAR":
            # Returns (Score, Action, Target, Owner)
            bids.append(self._perspective_newton(grid, agent_rc, rows, cols))

        # Euclid (Geometry/Completion)
        bids.append(self._perspective_euclid(grid, rows, cols))

        # Skinner (Causality/Probing)
        bids.append(self._perspective_skinner(grid, agent_rc, rows, cols))

        # Filter None
        valid_bids = [b for b in bids if b[1] is not None]
        if not valid_bids:
            return self._perspective_chaos()

        weighted_bids = []
        for score, action, target, owner in valid_bids:
            w_score = score * GlobalNexus.PERSPECTIVE_WEIGHTS[owner]
            # Safety: Don't walk into known walls (unless Recovery)
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            weighted_bids.append((w_score, action, target, owner))

        weighted_bids.sort(key=lambda x: x[0], reverse=True)
        score, action, target, owner = weighted_bids[0]

        # 6. EXECUTION
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            # Safety: Don't walk into known walls (unless Recovery overrides)
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0
            # Meta-Weight Application
            w_score = score * GlobalNexus.PERSPECTIVE_WEIGHTS[owner]

            # Recursive Critique: Safety
            if owner in ["Newton", "Fusion"] and target:
                 if target in self.walls: w_score *= 0.0 # Veto known walls

            weighted_bids.append((w_score, action, target, owner))

        weighted_bids.sort(key=lambda x: x[0], reverse=True)
        best_bid = weighted_bids[0]
        score, action, target, owner = best_bid

        # 8. EXECUTION
        # 7. EXECUTION
        if target:
            val = 1
            if len(target) > 2: val = target[2]
            else:
                try:
                    gval = grid[target[0], target[1]]
                    val = gval if gval != 0 else 1
                except: pass
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
        for (r,c), label in self.vision_targets:
            if (r,c) not in self.interactions:
                # SEMANTIC MAPPING
                val = 1 # Default Blue
                label_lower = label.lower()
                for name, v_int in COLOR_MAP.items():
                    if name in label_lower:
                        val = v_int
                        break
                return (0.97, self.ACT_CLICK, (r,c,val), f"Oracle({label})")
        return (0.0, None, None, "Oracle")

    def _perspective_newton(self, grid, agent_rc, rows, cols, boost=1.0):
        # Bids on visually detected objects
        for (r,c), label in self.vision_targets:
            if (r,c) not in self.interactions:
                # High base confidence for VLM detection
                return (0.97, self.ACT_CLICK, (r,c,1), f"Oracle({label})")
        return (0.0, None, None, "Oracle")

    def _perspective_newton(self, grid, agent_rc, rows, cols, boost=1.0):
        self.last_perspective = owner
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
        # 4. RECURSIVE CRITIQUE (The Cortex)
        # Newton validates targets for reachability
        vetted_bids = []
        for score, action, target, owner in valid_bids:
            final_score = score

            # Critique: "Can I reach this click target?"
            if target and agent_rc and owner in ["Euclid", "Skinner"]:
                path = self._astar_soft(grid, agent_rc, target, rows, cols)
                if not path:
                    # VETO: Target is behind a wall or unreachable
                    final_score *= 0.1

            # Critique: "Am I stuck?" -> Boost Skinner
            if owner == "Skinner" and self.stuck_counter > 2:
                final_score *= 2.0

            vetted_bids.append((final_score, action, target, owner))

        # 5. SELECTION & SIDE EFFECTS
        vetted_bids.sort(key=lambda x: x[0], reverse=True)
        best_bid = vetted_bids[0]
        score, action, target, owner = best_bid

        # Conflict Resolution: If tie between Pattern & Nav, prioritize Nav (Survival)
        if len(vetted_bids) > 1:
            runner = vetted_bids[1]
            if (score - runner[0] < 0.05) and (owner == "Euclid") and (runner[3] == "Newton"):
                best_bid = runner
                score, action, target, owner = best_bid

        # Apply Side Effects (Memory Update) ONLY for the winner
        if target:
            if action == self.ACT_CLICK:
                self.interactions.add(target)
                self._set_click_payload(target) # Sets self.action_data
            elif action == self.ACT_USE:
                self.interactions.add(target)

        self.last_action = action
        return action

    # --- PERSPECTIVES ---

    def _perspective_newton(self, grid, agent_rc, rows, cols):
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
        # Vector
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            target, color = ext
            return (0.92 * boost, self.ACT_CLICK, (target[0], target[1], color), "Euclid")
        # Symmetry (RESTORED)
        sym = self._detect_symmetry_completion(grid, rows, cols)
        if sym:
            target, color = sym
            return (0.90 * boost, self.ACT_CLICK, (target[0], target[1], color), "EuclidSym")
        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols, boost=1.0):
        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols, boost=1.0):
            # Allow push (cost=5)
            path = self._astar(grid, agent_rc, t_rc, rows, cols, allow_push=True)
            path = self._astar(grid, agent_rc, t_rc, rows, cols, allow_push=True)
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
        """Interaction: Probing."""
        # Local Use
        """Navigation: Soft A* with Dynamic Obstacles"""
        if not agent_rc: return (0.0, None, None, "Newton")

        targets = self._scan_targets(grid, agent_rc)
        if not targets:
            # Frontier Logic
            frontier = self._find_frontier(grid, agent_rc, rows, cols)
            if frontier: return (0.6, frontier[0], None, "Newton")
            return (0.0, None, None, "Newton")

        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue
            path = self._astar_soft(grid, agent_rc, t_rc, rows, cols)
            if path: return (0.95, path[0], None, "Newton")
            else: self.bad_goals.add(t_rc)

        return (0.0, None, None, "Newton")

    def _perspective_euclid(self, grid, rows, cols):
        """Geometry: Vector Extension + Symmetry"""
        # 1. Vector Extension
        ext = self._find_vector_extension(grid, rows, cols)
        if ext:
            return (0.9, self.ACT_CLICK, ext, "Euclid")

        # 2. Symmetry Completion (v4.0 Feature)
        sym_target = self._detect_symmetry_completion(grid, rows, cols)
        if sym_target:
             return (0.85, self.ACT_CLICK, sym_target, "Euclid")

        return (0.0, None, None, "Euclid")

    def _perspective_skinner(self, grid, agent_rc, rows, cols):
        """Causality: Weighted Interaction"""
        # 1. Local Interaction (Spacebar)
        if agent_rc:
            adj = self._scan_adjacent(grid, agent_rc, rows, cols)
            for r, c in adj:
                val = grid[r, c]
                if val != 0 and (r,c) not in self.interactions:
                    score = 0.8 * GlobalNexus.USE_SCORES[val] * boost
                    return (score, self.ACT_USE, (r,c), "Skinner")
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
                         score = 0.6 * GlobalNexus.CLICK_SCORES[val] * boost
                         return (score, self.ACT_CLICK, (r,c, val), "Skinner")
                         score = 0.6 * GlobalNexus.CLICK_SCORES[val]
                         return (score, self.ACT_CLICK, (r,c, val), "Skinner")
                    score = 0.8 * GlobalNexus.CAUSAL_SCORES[val]
                    return (score, self.ACT_USE, (r,c), "Skinner")

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
            if dist > 1 and t_rc not in self.interactions:
            if dist > 1 and (t_rc, self.ACT_USE) not in self.interactions:
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
            if self.stuck_counter > 5: return (3.0, self._perspective_chaos(), None, "Recovery")
            if self.stuck_counter > 5:
        # COSMOS FIX: Triggers on Global Stasis (stuck_counter > 2)
        if self.stuck_counter > 2:
            if self.stuck_counter > 5:
                # CHAOS INJECTION (Priority 3.0)
                return (3.0, self._perspective_chaos(), None, "Recovery")
            # Try interaction
        # NEW: Triggers on stuck_counter, which now respects global stasis
        if self.stuck_counter > 2:
            if self.stuck_counter > 5:
                # BREAK THE LOOP: Teleport via Chaos
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
        prev_hash = self.last_grid.tobytes() if self.last_grid is not None else b""
        if self.last_action in [self.ACT_CLICK, self.ACT_USE] and self.last_grid is not None:
             success = (current_hash != prev_hash)
             val = self.last_target_val
             if val is not None:
                 target_dict = GlobalNexus.USE_SCORES if self.last_action == self.ACT_USE else GlobalNexus.CLICK_SCORES
                 target_dict[val] *= (1.2 if success else 0.8)

        state_changed = (current_hash != prev_hash)
        moved = (agent_rc != self.last_pos) if self.last_pos is not None and agent_rc is not None else False
        if not state_changed and not moved: self.stuck_counter += 1
        else: self.stuck_counter = 0

                    # Boost score if this object type has reacted before
                    score = 0.7 + (self.causal_history[val] * 0.2)
                    return (min(0.99, score), self.ACT_USE, (r,c), "Skinner")

        # 2. Global Clicking
        unique, counts = np.unique(grid, return_counts=True)
        indices = np.argsort(counts)
        sorted_vals = unique[indices]
        self.action_data = {} # Reset
        # Reset payload every tick to prevent leakage
        self.action_data = {}

        if latest.state == GameState.NOT_PLAYED or not latest.frame:
            return GameAction.RESET

        try:
            return self._kull_logic(latest)
        except Exception:
            return self._random_move()

    def _kull_logic(self, latest: FrameData) -> GameAction:
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape
        current_hash = np.sum(grid)
        current_bytes = grid.tobytes()

        # 1. LOCATE AGENT (Robust)
        agent_rc = self._locate_agent(latest, grid)

        # 2. DYNAMIC IDENTITY (If lost, find who moved)
        if self.mode == "AVATAR" and self.last_grid is not None and self.last_pos is not None:
            prev_rc = self.last_pos
            prev_bytes = self.last_state[2] if self.last_state else b""

            # If world changed but we are lost/static
            grid_changed = (current_bytes != prev_bytes)

            if grid_changed:
                if agent_rc is None or agent_rc == prev_rc:
                     new_color = self._detect_moving_color(self.last_grid, grid, prev_rc)
                     if new_color is not None:
                         self.agent_color = new_color
                         agent_rc = self._locate_agent(latest, grid) # Retry

        # --- L0: WIGGLE CALIBRATION (Steps 1-4) ---
        if self.mode == "CALIBRATING":
            # Check results of last move
            if self.last_state:
                prev_rc = self.last_pos
                prev_bytes = self.last_state[2]
                grid_changed = (current_bytes != prev_bytes)

                # If we moved or grid changed (physics response)
                if (agent_rc and prev_rc and agent_rc != prev_rc) or grid_changed:
                    self.mode = "AVATAR"
                    self.plan.clear()

            # If still calibrating
            if self.mode == "CALIBRATING":
                step_idx = self.boot_steps
                if step_idx >= len(self.calibration_moves):
                    # Ran out of wiggles -> God Mode
                    self.mode = "SCIENTIST"
                else:
                    # Execute next wiggle
                    move = self.calibration_moves[step_idx]
                    self.boot_steps += 1
                    self._update_state(grid, agent_rc, current_hash, current_bytes)
                    self.last_action = move
                    return move

        # --- L1: PHYSICS UPDATE (Avatar) ---
        # Inventory Change -> Doors Open -> Clear Walls
            return self._polymathic_choose(latest)
        except Exception as e:
            # Fallback for robustness (never freeze)
            return self._random_move()

    def _polymathic_choose(self, latest: FrameData) -> GameAction:
        grid = np.array(latest.frame[-1])
        rows, cols = grid.shape
        agent_rc = self._locate_agent(latest, grid)
        current_hash = np.sum(grid)

        # --- L0: PHYSICS CALIBRATION ---
        if self.mode == "CALIBRATING":
            self.boot_steps += 1

            # 1. Try to Move (Generates delta)
            if self.boot_steps == 1:
                move = self._find_open_move(grid, agent_rc)
                self.last_state = (agent_rc, current_hash, grid.tobytes())
                self.last_action = move
                return move

            # 2. Analyze Result
            prev_rc, prev_hash, prev_bytes = self.last_state

            # Did we move? (Body exists)
            pos_changed = (agent_rc and prev_rc and agent_rc != prev_rc)
            # Use raw bytes for grid change, as sum() is invariant to movement
            grid_changed = (grid.tobytes() != prev_bytes)

            # Dynamic Color Logic: If grid changed but agent didn't, we might be tracking wrong object
            if grid_changed and not pos_changed:
                 last_grid = np.frombuffer(prev_bytes, dtype=grid.dtype).reshape(grid.shape)
                 diff = grid != last_grid
                 changed_values = grid[diff]
                 candidates = [v for v in changed_values if v != 0]
                 if candidates:
                     from collections import Counter
                     most_common = Counter(candidates).most_common(1)
                     if most_common:
                         self.agent_color = most_common[0][0]
                         agent_rc = self._locate_agent(latest, grid)
                         pos_changed = (agent_rc != prev_rc)

            if pos_changed:
                self.mode = "AVATAR"
            elif grid_changed:
                # Grid changed -> Physics exist -> Avatar/Builder
                self.mode = "AVATAR"
            else:
                # Static. Likely a "Click to solve" puzzle (ft09/vc33)
                self.mode = "SCIENTIST"

        # --- L1: DYNAMIC PHYSICS (Avatar Mode) ---
        # If inventory changed (Key pickup), clear walls (Doors might open)
        if abs(current_hash - self.inventory_hash) > 0:
            self.walls.clear()
            self.plan.clear()
            self.bad_goals.clear()
        self.inventory_hash = current_hash

        # Collision Learning
        if self.mode == "AVATAR" and self.last_pos and self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
            # If we didn't move, and grid didn't change (no push)
            grid_same = (current_bytes == self.last_state[2])

            if agent_rc == self.last_pos and grid_same:
                self.stuck_counter += 1
                if self.stuck_counter > 0:
                    tr, tc = self._get_target(self.last_pos, self.last_action)
                    self.walls.add((tr, tc))
                    self.plan.clear()
            else:
                self.stuck_counter = 0

        # Update State
        self._update_state(grid, agent_rc, current_hash, current_bytes)

        # Decide
        action = self._decide_strategy(grid, agent_rc, rows, cols)
        self.last_action = action
        return action

    def _update_state(self, grid, rc, h, b):
        self.last_grid = grid.copy()
        self.last_state = (rc, h, b)
        self.last_pos = rc

    def _decide_strategy(self, grid, agent_rc, rows, cols):
        if self.plan: return self.plan.popleft()
        # Collision Detection
        if self.mode == "AVATAR" and self.last_state:
            prev_rc, prev_hash = self.last_state
            if self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
                if prev_rc == agent_rc and agent_rc is not None and prev_hash == current_hash:
                    self.stuck_counter += 1
                    if self.stuck_counter > 0:
                        tr, tc = self._get_target(prev_rc, self.last_action)
                        self.walls.add((tr, tc))
                        self.plan.clear()
                else:
                    self.stuck_counter = 0

        self.last_state = (agent_rc, current_hash, grid.tobytes())

        # --- EXECUTION ---
        if self.plan:
            return self.plan.popleft()

        if self.mode == "AVATAR":
            return self._strategy_avatar(grid, agent_rc, rows, cols)
        else:
            return self._strategy_scientist(grid, rows, cols)

    def _strategy_avatar(self, grid, agent_rc, rows, cols):
        """Optimistic A* Navigation"""
        if not agent_rc: return self._random_move()

        # 1. Identify Goals (Rare objects)
        targets = self._scan_targets(grid, agent_rc)

        # 2. Pathfind (Optimistic)
        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue

            # OPTIMISTIC: Assume everything is walkable unless in self.walls
            # Hybrid Scientist/Builder Strategy
            return self._strategy_singularity(grid, rows, cols)

    def _strategy_avatar(self, grid, agent_rc, rows, cols):
        """Navigation Logic (A*)"""
        if not agent_rc: return self._random_move()

        # 1. Goal Inference (Rare objects)
        targets = self._scan_targets(grid, agent_rc)

        # 2. Pathfind
        for dist, t_rc in targets:
            if t_rc in self.walls or t_rc in self.bad_goals: continue

            path = self._astar(grid, agent_rc, t_rc, rows, cols)
            if path:
                self.plan.extend(path)
                return self.plan.popleft()
            else:
                self.bad_goals.add(t_rc)

        # 3. Frontier (Explore unknown)
        # 3. Frontier Exploration (If no goals)
        frontier = self._find_frontier(grid, agent_rc, rows, cols)
        if frontier:
            self.plan.extend(frontier)
            return self.plan.popleft()

        return self._random_move()

    def _strategy_scientist(self, grid, rows, cols):
        """Pattern Extension + Clicking"""
        # 1. Line Extension (ft09)
        ext = self._find_vector_extension(grid, rows, cols)
        if ext: return self._execute_click(ext)

        # 2. Rare Object Probe (vc33)
        targets = self._scan_targets(grid, (-1,-1)) # Scan all
        for _, t_rc in targets:
            if t_rc not in self.interactions:
                return self._execute_click(t_rc)

        # 3. Random
    def _strategy_singularity(self, grid, rows, cols):
        """
        Combined Scientist (Click) + Builder (Pattern Extension).
        Prioritizes:
        1. Extending Lines (ft09)
        2. Clicking Rare Objects (vc33)
        3. Random Probing
        """
        # --- PHASE 1: BUILDER (Vector Extrapolation) ---
        # Look for colored lines and click the empty spot at the end.
        extension_target = self._find_vector_extension(grid, rows, cols)
        if extension_target:
            return self._execute_click(extension_target)

        # --- PHASE 2: SCIENTIST (Rarity Probe) ---
        # Click non-background objects we haven't touched.
        unique, counts = np.unique(grid, return_counts=True)
        rare_indices = np.argsort(counts)
        sorted_vals = unique[rare_indices]

        for val in sorted_vals:
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

    def _run_handshake(self, grid, agent_rc):
    # --- SUBSTRATE & HANDSHAKE ---

    def _run_handshake(self, grid, agent_rc):
        # Did we learn anything?
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
                    if k not in self.control_map:
                        self.control_map[k] = v
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

        state_changed = (current_hash != np.sum(self.last_grid)) if self.last_grid is not None else True
        moved = (agent_rc != self.last_pos) if self.last_pos is not None and agent_rc is not None else False

        if not state_changed and not moved: self.stuck_counter += 1
        else: self.stuck_counter = 0

        # 2. PROGRESS-DECOUPLED STUCK DETECTION (The Fix)
        state_changed = (current_hash != np.sum(self.last_grid)) if self.last_grid is not None else True
        moved = (agent_rc != self.last_pos) if self.last_pos is not None and agent_rc is not None else False

        if not state_changed and not moved:
            self.stuck_counter += 1
            # Diagnostic Trace
            if self.stuck_counter >= 3:
                print(f"[v9.1] STUCK={self.stuck_counter} | Act={self.last_action} | Owner={self.last_perspective}")
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
        self.inventory_hash = np.sum(grid)

    # --- HELPERS ---
    def _detect_symmetry_completion(self, grid, rows, cols):
        # 1. Horizontal Mirror
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mirror_c = cols - 1 - c
                val_l = grid[r, c]
                val_r = grid[r, mirror_c]
                if val_l != 0 and val_r == 0:
                     if (r, mirror_c) not in self.interactions: return ((r, mirror_c), val_l)
                if val_r != 0 and val_l == 0:
                     if (r, c) not in self.interactions: return ((r, c), val_r)
        # 2. Vertical Mirror
        mid_r = rows // 2
        for r in range(mid_r):
            mirror_r = rows - 1 - r
            for c in range(cols):
                val_t = grid[r, c]
                val_b = grid[mirror_r, c]
                if val_t != 0 and val_b == 0:
                     if (mirror_r, c) not in self.interactions: return ((mirror_r, c), val_t)
                if val_b != 0 and val_t == 0:
                     if (r, c) not in self.interactions: return ((r, c), val_b)
        return None

                 # Identity Flush
                 if self.stuck_counter > 8:
                     self.agent_color = None

        self.last_pos = agent_rc
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

    def _find_center_of_change(self, prev, curr):
        if prev.shape != curr.shape: return None
        diff = prev != curr
        coords = np.argwhere(diff)
        if len(coords) == 0: return None
        return (int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1])))
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
    def _astar(self, grid, start, end, rows, cols):
        if not self.control_map: return None
                    # Weighted by Causal History
                    score = 0.5 + (self.causal_history[val] * 0.4)
                    return (min(0.99, score), self.ACT_CLICK, (r,c), "Skinner")

        return (0.0, None, None, "Skinner")

    def _perspective_chaos(self):
        return random.choice([self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT])

    # --- SUBSTRATE LOGIC ---

    def _update_physics(self, grid, agent_rc, current_hash):
        # 1. Identity Check
        if self.mode == "AVATAR" and self.last_grid is not None and self.last_pos is not None:
             if current_hash != self.last_state[1]:
                 # Causal Learning: Record what changed
                 diff = grid != self.last_grid
                 changed_vals = grid[diff]
                 for v in changed_vals:
                     if v != 0: self.causal_history[v] += 1.0 # Boost influence

                 # Identity Correction
                 if agent_rc is None or agent_rc == self.last_pos:
                      new_c = self._detect_moving_color(self.last_grid, grid, self.last_pos)
                      if new_c is not None: self.agent_color = new_c

        # 2. Wall/Goal Reset
        if abs(current_hash - self.inventory_hash) > 0:
            self.walls.clear()
            self.bad_goals.clear()
        self.inventory_hash = current_hash

        # 3. Collision
        if self.mode == "AVATAR" and self.last_pos and self.last_action in [self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT]:
             if agent_rc == self.last_pos and current_hash == self.last_state[1]:
                 self.stuck_counter += 1
                 if self.stuck_counter > 0:
                     tr, tc = self._get_target(self.last_pos, self.last_action)
                     self.walls.add((tr, tc))
             else:
                 self.stuck_counter = 0

        self.last_grid = grid.copy()
        self.last_state = (agent_rc, current_hash)
        self.last_pos = agent_rc

    def _run_calibration(self, grid, agent_rc):
        # If we moved or grid changed (physics response)
        if self.last_pos and agent_rc and agent_rc != self.last_pos:
            self.mode = "AVATAR"
            return self._omega_executive(FrameData(grid.tolist(), None, GameState.PLAYING))

        # Check grid change (for ls20 fix)
        if self.last_grid is not None:
             if not np.array_equal(grid, self.last_grid):
                 self.mode = "AVATAR" # World changed -> Physics
                 return self._omega_executive(FrameData(grid.tolist(), None, GameState.PLAYING))

        if self.calibration_queue:
            return self.calibration_queue.popleft()

        self.mode = "SCIENTIST"
        return self._omega_executive(FrameData(grid.tolist(), None, GameState.PLAYING))

    # --- ALGORITHMIC HELPERS ---

    def _detect_symmetry_completion(self, grid, rows, cols):
        """Detects missing pixels in Mirror Symmetry (Horizontal)."""
        mid = cols // 2
        for r in range(rows):
            for c in range(mid):
                mirror_c = cols - 1 - c
                val_l = grid[r, c]
                val_r = grid[r, mirror_c]

                # If Left exists and Right is empty -> Click Right
                if val_l != 0 and val_r == 0:
                    if (r, mirror_c) not in self.interactions:
                        return (r, mirror_c)
                # If Right exists and Left is empty -> Click Left
                if val_r != 0 and val_l == 0:
                    if (r, c) not in self.interactions:
                        return (r, c)
        return None

    def _find_vector_extension(self, grid, rows, cols):
        non_zeros = np.argwhere(grid != 0)
        if len(non_zeros) > 80: return None
        by_color = defaultdict(list)
        for r, c in non_zeros: by_color[grid[r,c]].append((r,c))

        for color, coords in by_color.items():
            if len(coords) < 2: continue
            coords.sort()
            for i in range(len(coords)-1):
                r1, c1 = coords[i]
                r2, c2 = coords[i+1]
                dr, dc = r2-r1, c2-c1
                if abs(dr) > 2 or abs(dc) > 2: continue

                # Project forward
                nr, nc = r2+dr, c2+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] == 0 and (nr, nc) not in self.interactions:
                        return (nr, nc)
        return None

    def _astar_soft(self, grid, start, end, rows, cols):
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
                    is_walk = (grid[nr, nc] == 0) or ((nr, nc) == end)
                    if (nr, nc) in self.walls: is_walk = False
                    if is_walk:
                        ng = steps + 1
                        if (nr, nc) not in best_g or ng < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = ng
                            h = abs(nr-end[0]) + abs(nc-end[1])
                            heapq.heappush(pq, (ng+h, ng, (nr, nc), path+[act]))
            if steps > 300: break

            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in self.walls: continue
                    val = grid[nr, nc]
                    cost = 1 if val == 0 or (nr, nc) == end else 5
                    new_g = steps + cost
                    if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                        best_g[(nr, nc)] = new_g
                        h = abs(nr - end[0]) + abs(nc - end[1])
                        heapq.heappush(pq, (new_g + h, new_g, (nr, nc), path + [act]))
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
            if val != 0 and prev_val == 0: cands.append(val)
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
        indices = np.argsort(counts)
        for i in indices:
            c, count = unique[i], counts[i]
            if count < 50:
                coords = np.argwhere(grid == c)
                    return self._execute_click((r, c))

        # --- PHASE 3: ENTROPY (Random) ---
        r = random.randint(0, rows-1)
        c = random.randint(0, cols-1)
        return self._execute_click((r, c))

    def _astar(self, grid, start, end, rows, cols):
        """Optimistic A* (Manhattan)"""
        pq = [(0, 0, start, [])]
        best_g = {start: 0}

        while pq:
            f, steps, curr, path = heapq.heappop(pq)
            if curr == end: return path
            if steps > 500: break

            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # OPTIMISM: If not a known wall, we try it.
                    if (nr, nc) not in self.walls:
                        new_g = steps + 1
                        if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = new_g
                            h = abs(nr - end[0]) + abs(nc - end[1])
                            heapq.heappush(pq, (new_g + h, new_g, (nr, nc), path + [act]))
        return None

    def _find_vector_extension(self, grid, rows, cols):
        """Finds line patterns and clicks the next empty spot."""
        non_zeros = np.argwhere(grid != 0)
        if len(non_zeros) > 80: return None

    def _find_vector_extension(self, grid, rows, cols):
        """
        Detects 2+ pixels of same color forming a line.
        Returns coordinate of the NEXT pixel in that line if it's 0 (empty).
        """
        # Scan for colored pixels
        non_zeros = np.argwhere(grid != 0)
        if len(non_zeros) > 60: return None # Too noisy/complex

        # Group by color
        by_color = {}
        for r, c in non_zeros:
            val = grid[r, c]
            if val not in by_color: by_color[val] = []
            by_color[val].append((r, c))

        for color, coords in by_color.items():
            if len(coords) < 2: continue
            # Check pairwise
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    r1, c1 = coords[i]
                    r2, c2 = coords[j]
                    dr, dc = r2 - r1, c2 - c1

                    # Must be adjacent or close (step 1 or 2)
                    if abs(dr) > 2 or abs(dc) > 2: continue

                    next_r, next_c = r2 + dr, c2 + dc
                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        if grid[next_r, next_c] == 0 and (next_r, next_c) not in self.interactions:
                            return (next_r, next_c)

            # Check every pair for a vector
            # Limit checks for speed (nearest neighbors)
            peers = coords[:12]
            for i in range(len(peers)):
                for j in range(i+1, len(peers)):
                    r1, c1 = peers[i]
                    r2, c2 = peers[j]

                    # Vector (dr, dc)
                    dr = r2 - r1
                    dc = c2 - c1

                    # Simplify vector (get unit step or exact step)
                    # For pattern completion, we usually want exact step repetition
                    step_r = dr
                    step_c = dc

                    # Project forward from (r2, c2)
                    next_r, next_c = r2 + step_r, c2 + step_c

                    if 0 <= next_r < rows and 0 <= next_c < cols:
                        # If empty and not clicked -> Candidate
                        if grid[next_r, next_c] == 0 and (next_r, next_c) not in self.interactions:
                            return (next_r, next_c)

                    # Project backward from (r1, c1)
                    prev_r, prev_c = r1 - step_r, c1 - step_c
                    if 0 <= prev_r < rows and 0 <= prev_c < cols:
                         if grid[prev_r, prev_c] == 0 and (prev_r, prev_c) not in self.interactions:
                            return (prev_r, prev_c)
        return None

    def _execute_click(self, rc):
        r, c = rc
        self.interactions.add((r, c))
        payload = {'x': int(c), 'y': int(r)}
        self.action_data = payload

        # SHIM: Attach data to the Enum Member
        payload = {'x': int(c), 'y': int(r)}

        try:
            self.ACT_CLICK.set_data(payload)
        except:
            pass
        return self.ACT_CLICK

    def _locate_agent(self, latest, grid):
        # 1. API
        if hasattr(latest, 'agent_pos') and latest.agent_pos and latest.agent_pos != (-1, -1):
            return (latest.agent_pos[1], latest.agent_pos[0])
        # 2. Dynamic Color
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0: return tuple(coords[0])
        # 3. Heuristic (Allow 0 if rare)
        unique, counts = np.unique(grid, return_counts=True)
        # Sort by rarity
        indices = np.argsort(counts)
        sorted_colors = unique[indices]
        sorted_counts = counts[indices]

        for color, count in zip(sorted_colors, sorted_counts):
            if count < 50: # Assume agent is small
                coords = np.argwhere(grid == color)
                if len(coords) > 0: return tuple(coords[0])
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
                d = abs(r - start_rc[0]) + abs(c - start_rc[1])
                targets.append((d, (r, c)))
        targets.sort()
        return targets

    def _detect_moving_color(self, prev_grid, curr_grid, last_pos):
        if prev_grid.shape != curr_grid.shape: return None
        diff = prev_grid != curr_grid
        if not np.any(diff): return None
        r, c = last_pos
        if not last_pos: return None
        r, c = last_pos
        diff = prev_grid != curr_grid
        if not np.any(diff): return None

        # Look around last position for new values
        neighbors = [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
        rows, cols = curr_grid.shape
        candidates = []
        for nr, nc in neighbors:
             if 0 <= nr < rows and 0 <= nc < cols:
                 val = curr_grid[nr, nc]
                 if val != 0 and val != prev_grid[nr, nc]: candidates.append(val)
                 if val != 0 and val != prev_grid[nr, nc]:
                     candidates.append(val)
        if candidates: return Counter(candidates).most_common(1)[0][0]
        return None

    def _find_frontier(self, grid, start, rows, cols):
        """BFS to unknown"""

        return self.ACT_CLICK

    def _astar(self, grid, start, end, rows, cols):
        """A* (Manhattan)"""
        pq = [(0, 0, start, [])]
        best_g = {start: 0}

        while pq:
            f, steps, curr, path = heapq.heappop(pq)
            if curr == end: return path
            if steps > 500: break

            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    # Walkable if Air (0) OR Goal (Walking onto key/door/button is good)
                    is_walkable = (grid[nr, nc] == 0) or ((nr, nc) == end)
                    if (nr, nc) in self.walls: is_walkable = False

                    if is_walkable:
                        new_g = steps + 1
                        if (nr, nc) not in best_g or new_g < best_g[(nr, nc)]:
                            best_g[(nr, nc)] = new_g
                            h = abs(nr - end[0]) + abs(nc - end[1])
                            heapq.heappush(pq, (new_g + h, new_g, (nr, nc), path + [act]))
        return None

    def _find_frontier(self, grid, start, rows, cols):
        """BFS to unknown space"""
        q = deque([(start, [])])
        seen = {start}
        steps = 0
        while q:
            steps += 1;
            if steps > 200: break
            curr, path = q.popleft(); r, c = curr
            for act, (dr, dc) in self.control_map.items():
            steps += 1
            if steps > 200: break
            curr, path = q.popleft()
            r, c = curr
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]
            for (dr, dc), act in moves:
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
        try: self.ACT_CLICK.set_data(payload)
        except: pass
    def _set_click_payload(self, rc):
        r, c = rc
        payload = {'x': int(c), 'y': int(r)}
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
            moves = [((-1,0), self.ACT_UP), ((1,0), self.ACT_DOWN),
                     ((0,-1), self.ACT_LEFT), ((0,1), self.ACT_RIGHT)]

            for (dr, dc), act in moves:
                nr, nc = r+dr, c+dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) in self.walls or grid[nr, nc] != 0: continue
                    if (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append(((nr, nc), path+[act]))

            # If deep enough, consider it a frontier
            if len(path) > 8: return path
        return None

    def _locate_agent(self, latest, grid):
        """Robust Location"""
        # 0. Learned Color
        if self.agent_color is not None:
             coords = np.argwhere(grid == self.agent_color)
             if len(coords) > 0:
                 return tuple(coords[0])

        if hasattr(latest, 'agent_pos') and latest.agent_pos and latest.agent_pos != (-1, -1):
            return (latest.agent_pos[1], latest.agent_pos[0])

        # Visual Scan (Colors 1, 2, 8 are common agents)
        for color in [1, 2, 8, 3]:
            matches = np.argwhere(grid == color)
            if len(matches) == 1: return tuple(matches[0])

        unique, counts = np.unique(grid, return_counts=True)
        sorted_indices = np.argsort(counts)
        sorted_colors = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]

        for color, count in zip(sorted_colors, sorted_counts):
            if color == 0 and count > 100: continue
            if 0 < count < 50:
                coords = np.argwhere(grid == color)
                if len(coords) > 0: return tuple(coords[0])
        return None

    def _scan_targets(self, grid, start_rc):
        unique, counts = np.unique(grid, return_counts=True)
        rare = unique[(counts < 40) & (unique != 0)]
        targets = []
        matches = np.argwhere(np.isin(grid, rare))
        for r, c in matches:
            if (r, c) != start_rc:
                d = abs(r - start_rc[0]) + abs(c - start_rc[1])
                targets.append((d, (r, c)))
        targets.sort()
        return targets

    def _get_target(self, rc, action):
        r, c = rc
        if action == self.ACT_UP: return r-1, c
        if action == self.ACT_DOWN: return r+1, c
        if action == self.ACT_LEFT: return r, c-1
        if action == self.ACT_RIGHT: return r, c+1
        return r, c

    def _find_open_move(self, grid, rc):
        if not rc: return self.ACT_RIGHT
        r, c = rc
        h, w = grid.shape
        # Prioritize Right, then others
        if c < w-1 and grid[r, c+1] == 0: return self.ACT_RIGHT
        if r < h-1 and grid[r+1, c] == 0: return self.ACT_DOWN
        if c > 0 and grid[r, c-1] == 0: return self.ACT_LEFT
        if r > 0 and grid[r-1, c] == 0: return self.ACT_UP
        return self.ACT_RIGHT

    def _random_move(self):
        return random.choice([self.ACT_UP, self.ACT_DOWN, self.ACT_LEFT, self.ACT_RIGHT])
