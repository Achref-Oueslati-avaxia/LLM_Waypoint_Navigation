#!/usr/bin/env python3
"""
LLM Mission Planner — Outlander Robot
======================================
Converts natural language mission commands into waypoint.json entries.

Behavior:
- Relative motion commands:
    * move forward/backward
    * turn left/right
  default to:
    * mode = "straight"
    * priority = false

- Named places:
    * home, base, bridge, etc.
  default to:
    * mode = "nav2"
    * priority = false

- Explicit user overrides are supported:
    * "priority", "urgent" -> priority = true
    * "use straight", "mode straight" -> mode = straight
    * "use nav2", "mode nav2" -> mode = nav2

- Stop commands:
    * handled immediately in Python
    * converted into a priority waypoint at the current pose with mode nav2 to cancel active goals
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import select
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
import threading
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
from std_msgs.msg import String, Bool
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import math
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from rclpy.time import Time
try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


# =============================================================================
# File paths
# =============================================================================

WORKSPACE = "/jazzy_ws"

WAYPOINTS_FILE = os.path.join(
    WORKSPACE,
    "src",
    "outlander_navigation",
    "config",
    "waypoints.json",
)

NAMED_DESTINATIONS_FILE = os.path.join(
    WORKSPACE,
    "src",
    "outlander_navigation",
    "config",
    "named_destinations.yaml",
)

# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an accurate robot mission interpreter for the Outlander robot.

Your job is NOT to compute map coordinates.
Your job is ONLY to convert the user's natural language command into a list of safe action steps.

Return ONLY valid JSON matching this exact format:
{
  "understood": "one short sentence summarizing the command",
  "steps": [
    {
      "action": "move|turn|goto|stop|resume",
      "direction": "forward|backward|left|right",
      "distance_m": 0.0,
      "angle_deg": 0.0,
      "target_name": "",
      "count": 1,
      "mode": "straight|nav2",
      "priority": false
    }
  ],
  "notes": ""
}

STRICT PARSING & SAFETY RULES:
1. NO COORDINATES:
   - Never output x, y, or yaw. Never invent map coordinates.

2. ACTION SELECTION:
   - Use action "move" for relative forward/backward movement with distance (e.g., "move 2m", "navigate forward 5m", "drive ahead 3m").
   - Use action "turn" for in-place rotations with angle (e.g., "turn right 90 deg", "spin left 45").
   - Use action "goto" ONLY if the command explicitly names a location present in known_places (e.g., "go to bridge", "navigate to home").
   - NEVER use action "goto" or populate "target_name" unless that location name literally appears in the operator command.

3. EXECUTION MODES ("mode"):
   - Defaults: "move" and "turn" default to "straight"; "goto" defaults to "nav2".
   - If the operator explicitly requests a mode (e.g., "in straight mode", "use nav2", "mode straight"), set "mode" accordingly for that step.

4. SAFETY & EDGE CASES:
   - If the command is unclear, ambiguous, or unsafe, set "steps": [] and explain why in "notes".
   - Keep the steps list as short and concise as possible.

5. JSON FORMATTING:
   - Use lowercase JSON booleans (true/false) only.
   - "count" must be an integer >= 1.
   - Output ONLY the requested raw JSON structure.
"""


# =============================================================================
# Synonym normalization
# =============================================================================

ACTION_NORMALIZATION = {
    "travel": "move",
    "advance": "move",
    "proceed": "move",
    "shift": "move",

    "rotate": "turn",
    "pivot": "turn",
    "swivel": "turn",
    "face": "turn",

    "head to": "goto",
    "proceed to": "goto",
    "navigate to": "goto",
    "approach": "goto",

    "halt": "stop",
    "cancel": "stop",
    "abort": "stop",
    "freeze": "stop",
}

DIRECTION_NORMALIZATION = {
    "ahead": "forward",
    "onward": "forward",
    "frontward": "forward",
    "straight ahead": "forward",

    "reverse": "backward",
    "rearward": "backward",
    "back": "backward",
    "astern": "backward",

    "port": "left",
    "counterclockwise": "left",
    "sinistral": "left",

    "starboard": "right",
    "clockwise": "right",
    "dextral": "right",
}

MODE_NORMALIZATION = {
    "direct": "straight",
    "linear": "straight",
    "unswerving": "straight",
    "point to point": "straight",

    "autonomous": "nav2",
    "map based": "nav2",
    "intelligent routing": "nav2",
    "path planned": "nav2",
    "pathplanned": "nav2",
}


# =============================================================================
# Allowed values
# =============================================================================

ALLOWED_ACTIONS = {"move", "turn", "goto", "stop", "resume"}
MOVE_DIRECTIONS = {"forward", "backward"}
TURN_DIRECTIONS = {"left", "right"}
ALLOWED_MODES = {"straight", "nav2"}

STOP_COMMAND_RE = re.compile(
    r"\b(?:stop|halt|abort|cancel|e[- ]?stop|emergency\s+stop)\b",
    re.IGNORECASE,
)
RESUME_COMMAND_RE = re.compile(r"\b(?:resume|continue|proceed)\b", re.IGNORECASE)

# =============================================================================
# Backend implementations
# =============================================================================

class _GroqBackend:
    """Groq Cloud backend."""

    DEFAULT_MODEL = "llama-3.1-70b-versatile"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from groq import Groq  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "groq package not installed. "
                "Run: pip install groq --break-system-packages"
            ) from e

        self._client = Groq(api_key=api_key)
        self._model = model or self.DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content.strip()


class _OllamaBackend:
    """Ollama backend for local/offline use."""

    DEFAULT_MODEL = "qwen2.5:7b"

    def __init__(self, host: str, model: str) -> None:
        try:
            import ollama as _ollama  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "ollama package not installed. "
                "Run: pip install ollama --break-system-packages"
            ) from e

        self._ollama = _ollama
        self._host = host or "http://localhost:11434"
        self._model = model or self.DEFAULT_MODEL
        self._client = self._ollama.Client(host=self._host)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        response = self._client.chat(
            model=self._model,
            format="json",
            options={
                "num_predict": max_tokens,
                "temperature": 0.0,
                "top_p": 0.1,
            },
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response["message"]["content"].strip()


class _OpenRouterBackend:
    """OpenRouter backend."""

    DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        import requests  # type: ignore
        self._requests = requests
        self._api_key = api_key
        self._model = model or self.DEFAULT_MODEL

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/outlander-robot",
        }
        body = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 0.1,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        resp = self._requests.post(self.API_URL, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class _GeminiBackend:
    """Google AI Studio backend."""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai --break-system-packages"
            ) from e

        genai.configure(api_key=api_key)
        self._model_obj = genai.GenerativeModel(model or self.DEFAULT_MODEL)

    def complete(self, system: str, user: str, max_tokens: int) -> str:
        full_prompt = f"{system}\n\n{user}"
        response = self._model_obj.generate_content(
            full_prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 0.1,
            },
        )
        return response.text.strip()


# =============================================================================
# Backend factory
# =============================================================================

def _build_backend(backend: str, model: str):
    backend = backend.lower().strip()

    if backend == "groq":
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in environment / .env file")
        return _GroqBackend(api_key=key, model=model)

    if backend == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return _OllamaBackend(host=host, model=model)

    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment / .env file")
        return _OpenRouterBackend(api_key=key, model=model)

    if backend == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set in environment / .env file")
        return _GeminiBackend(api_key=key, model=model)

    raise ValueError(
        f"Unknown backend '{backend}'. "
        "Choose: groq | ollama | openrouter | gemini"
    )


# =============================================================================
# Main ROS 2 node
# =============================================================================

class LLMMissionPlanner(Node):
    def __init__(self) -> None:
        super().__init__("llm_mission_planner")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.declare_parameter("llm_backend", "ollama")
        self.declare_parameter("llm_model", "qwen2.5:7b")
        self.declare_parameter("max_tokens", 512)
        self.declare_parameter("command_topic", "/robot/mission_command")
        self.declare_parameter("status_topic", "/robot/mission_status")
        self.declare_parameter("waypoints_file", WAYPOINTS_FILE)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("next_waypoint_id", 100)
        self.declare_parameter("named_destinations_file", NAMED_DESTINATIONS_FILE)
        self.declare_parameter("save_waypoints", "auto")

        self._max_tokens = int(self.get_parameter("max_tokens").value)
        self._wp_file = str(self.get_parameter("waypoints_file").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._named_destinations_file = str(self.get_parameter("named_destinations_file").value)
        self._waypoints_path = Path(self._wp_file)
        self._waypoints_file_existed = self._waypoints_path.exists()
        self._waypoints_backup_text = self._waypoints_path.read_text() if self._waypoints_file_existed else ""

        self._save_waypoints = self._resolve_save_waypoints_mode(
            str(self.get_parameter("save_waypoints").value)
        )
        self.get_logger().info(
            f"Waypoints session mode: {'SAVE' if self._save_waypoints else 'NO-SAVE'}"
        )

        self._next_id = self._recover_next_id()
        self._named_destinations = self._load_named_destinations()

        backend_name = str(self.get_parameter("llm_backend").value)
        model_name = str(self.get_parameter("llm_model").value)

        try:
            self._backend = _build_backend(backend_name, model_name)
            self.get_logger().info(
                f"LLM backend: {backend_name} | model: {model_name or '(backend default)'}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to init LLM backend: {e}")
            raise

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        cmd_topic = str(self.get_parameter("command_topic").value)
        status_topic = str(self.get_parameter("status_topic").value)

        self._cmd_sub = self.create_subscription(String, cmd_topic, self._on_command, 10)
        self._status_pub = self.create_publisher(String, status_topic, 10)
        self.robot_busy = False

        self.create_subscription(
            Bool,
            "/robot/mission_busy",
            self._busy_callback,
            10,
        )
        self._processing = False
        self.get_logger().info(f"LLM Mission Planner ready | listening on {cmd_topic}")

        self._executor_pool = ThreadPoolExecutor(max_workers=2)
        self._chat_history = deque(maxlen=5)

        self._cli_thread = threading.Thread(target=self._terminal_input_loop, daemon=True)
        self._cli_thread.start()

    def _load_named_destinations(self) -> dict:
        path = Path(self._named_destinations_file)
        if not path.exists():
            raise RuntimeError(f"Named destinations file not found: {path}")

        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Install python3-yaml / pyyaml.")

        raw = path.read_text().strip()
        if not raw:
            raise RuntimeError(f"Named destinations file is empty: {path}")

        data = yaml.safe_load(raw)
        if not isinstance(data, dict):
            raise RuntimeError(f"Named destinations file must contain a mapping: {path}")

        if "named_destinations" in data and isinstance(data["named_destinations"], dict):
            data = data["named_destinations"]

        cleaned: dict[str, dict] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                cleaned[str(key).strip().lower()] = value

        if not cleaned:
            raise RuntimeError(f"No valid named destinations found in: {path}")

        return cleaned

    def _resolve_named_destination(self, name: str) -> Optional[dict]:
        key = re.sub(r"\s+", "_", str(name).strip().lower())
        return self._named_destinations.get(key)

    def get_current_robot_pose(self):
        """Fetches the live map -> base_link transform."""
        try:
            # Get the latest available transform
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
            
            x = t.transform.translation.x
            y = t.transform.translation.y
            
            # Convert quaternion to yaw
            q = t.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            return x, y, yaw
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f"Could not get current robot pose via TF2: {e}")
            return None, None, None
    
    
    def _terminal_input_loop(self) -> None:
        print("\n" + "="*50)
        print(" Interactive Terminal Ready. Type commands directly below:")
        print(" Example: 'move forward 2m' or 'head to home'")
        print("="*50 + "\n")

        while rclpy.ok():
            try:
                raw_cmd = input("outlander_cmd> ").strip()
                if raw_cmd:
                    msg = String()
                    msg.data = raw_cmd
                    self._on_command(msg)
            except (EOFError, KeyboardInterrupt):
                break

    def _get_robot_pose(self) -> Optional[dict]:
        try:
            tf = self._tf_buffer.lookup_transform(self._map_frame, self._base_frame, Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            return {"x": round(t.x, 3), "y": round(t.y, 3), "yaw": round(yaw, 3)}
        except Exception:
            return None

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _normalize_command_text(self, command: str) -> str:
        text = command.lower().replace("_", " ").replace("-", " ")

        for phrase, canon in sorted(ACTION_NORMALIZATION.items(), key=lambda x: -len(x[0])):
            text = re.sub(rf"\b{re.escape(phrase)}\b", canon, text)

        for phrase, canon in sorted(DIRECTION_NORMALIZATION.items(), key=lambda x: -len(x[0])):
            text = re.sub(rf"\b{re.escape(phrase)}\b", canon, text)

        for phrase, canon in sorted(MODE_NORMALIZATION.items(), key=lambda x: -len(x[0])):
            text = re.sub(rf"\b{re.escape(phrase)}\b", canon, text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _command_requires_pose(self, command: str) -> bool:
        text = command.lower()
        if self._is_stop_command(text):
            return False

        relative_tokens = (
            r"\bmove\b",
            r"\bturn\b",
            r"\bforward\b",
            r"\bbackward\b",
            r"\bleft\b",
            r"\bright\b",
            r"\bmeters?\b",
            r"\bmetres?\b",
            r"\bdegrees?\b",
            r"\bdeg\b",
        )
        return any(re.search(pat, text) for pat in relative_tokens)

    def _resolve_save_waypoints_mode(self, raw_mode: str) -> bool:
        mode = str(raw_mode).strip().lower()

        if mode in ("true", "1", "yes", "y", "on"):
            return True
        if mode in ("false", "0", "no", "n", "off"):
            return False

        if mode != "auto":
            self.get_logger().warn(
                f"Unknown save_waypoints mode '{raw_mode}', defaulting to NO-SAVE"
            )
            return False

        if not sys.stdin.isatty():
            return False

        try:
            print(
                "Save waypoints for this session? [y/N] "
                "(auto-timeout 10s): ",
                end="",
                flush=True,
            )
            ready, _, _ = select.select([sys.stdin], [], [], 10.0)
            if not ready:
                print("\nNo input received; defaulting to NO-SAVE")
                return False

            answer = sys.stdin.readline().strip().lower()
            return answer in ("y", "yes", "true", "1")
        except Exception:
            return False

    def _load_waypoints(self) -> list:
        try:
            p = Path(self._wp_file)
            if not p.exists():
                return []
            raw = p.read_text().strip()
            if not raw:
                return []
            data = json.loads(raw)
            return data.get("waypoints", []) if isinstance(data, dict) else []
        except Exception:
            return []

    def _recover_next_id(self) -> int:
        try:
            fallback = int(self.get_parameter("next_waypoint_id").value)
            waypoints = self._load_waypoints()

            ids = []
            for wp in waypoints:
                raw_id = wp.get("id", "")
                if str(raw_id).isdigit():
                    ids.append(int(raw_id))

            return (max(ids) + 1) if ids else fallback
        except Exception:
            return int(self.get_parameter("next_waypoint_id").value)

    # def _append_waypoints(self, new_wps: list) -> None:
    #     path = self._waypoints_path
    #     path.parent.mkdir(parents=True, exist_ok=True)

    #     existing = self._load_waypoints()
    #     existing.extend(new_wps)

    #     tmp_handle = None
    #     tmp_name = None
    #     try:
    #         tmp_handle = tempfile.NamedTemporaryFile(
    #             mode="w",
    #             encoding="utf-8",
    #             delete=False,
    #             dir=str(path.parent),
    #             prefix=f".{path.stem}.",
    #             suffix=".tmp",
    #         )
    #         tmp_name = tmp_handle.name
    #         json.dump({"waypoints": existing}, tmp_handle, indent=2)
    #         tmp_handle.flush()
    #         os.fsync(tmp_handle.fileno())
    #         tmp_handle.close()

    #         os.replace(tmp_name, path)
    #         self.get_logger().info(f"Wrote {len(new_wps)} waypoint(s) → {self._wp_file}")
    #     finally:
    #         try:
    #             if tmp_handle is not None and not tmp_handle.closed:
    #                 tmp_handle.close()
    #         except Exception:
    #             pass
    #         try:
    #             if tmp_name and os.path.exists(tmp_name):
    #                 os.unlink(tmp_name)
    #         except Exception:
    #             pass
    
    def _append_waypoints(self, new_wps: list) -> None:
            path = self._waypoints_path
            path.parent.mkdir(parents=True, exist_ok=True)

            existing = self._load_waypoints()
            existing.extend(new_wps)

            with open(path, "w", encoding="utf-8") as f:
                json.dump({"waypoints": existing}, f, indent=2)

            self.get_logger().info(f"Wrote {len(new_wps)} waypoint(s) → {self._wp_file}")
            
        
    def _safe_parse_json(self, raw: str) -> dict:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response text")

        cleaned = match.group(0)
        cleaned = re.sub(r"\bTrue\b", "true", cleaned)
        cleaned = re.sub(r"\bFalse\b", "false", cleaned)
        cleaned = re.sub(r",\s*\}", "}", cleaned)
        cleaned = re.sub(r",\s*\]", "]", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            data = ast.literal_eval(cleaned)
            if isinstance(data, dict):
                return data
            raise

    def _publish_status(self, msg: str) -> None:
        self._status_pub.publish(String(data=msg))
        self.get_logger().info(f"Status → MQTT: {msg}")

    def _extract_command_overrides(self, command: str) -> tuple[Optional[str], Optional[bool]]:
        text = command.lower()

        mode_override: Optional[str] = None
        priority_override: Optional[bool] = None

        if re.search(r"\b(?:use|mode|in)\s+straight\b", text) or re.search(r"\bstraight\s+mode\b", text):
            mode_override = "straight"
        elif re.search(r"\b(?:use|mode|in)\s+nav2\b", text) or re.search(r"\bnav2\s+mode\b", text):
            mode_override = "nav2"
        elif re.search(r"\bstraight\b", text):
            mode_override = "straight"
        elif re.search(r"\bnav2\b", text):
            mode_override = "nav2"

        if re.search(r"\b(?:no|without|non[- ]?)\s+priority\b", text):
            priority_override = False

        elif re.search(r"\bpriority\s+(?:true|1|yes|on|high)\b", text):
            priority_override = True

        elif re.search(
            r"\b("
            r"priority|"
            r"urgent|urgently|"
            r"high priority|"
            r"immediately|"
            r"right now|"
            r"right away|"
            r"asap|"
            r"as soon as possible|"
            r"at once|"
            r"without delay|"
            r"straight away|"
            r"now"
            r")\b",
            text,
        ):
            priority_override = True
            
        return mode_override, priority_override

    def _parse_optional_mode(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        s = str(value).strip().lower()
        return s if s in ALLOWED_MODES else None

    def _parse_optional_priority(self, value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y", "t", "on"):
            return True
        if s in ("false", "0", "no", "n", "f", "off"):
            return False
        return None

    def _default_mode_for_action(self, action: str) -> str:
        if action in ("move", "turn"):
            return "straight"
        if action == "goto":
            return "nav2"
        return "straight"

    def _is_stop_command(self, command: str) -> bool:
        return bool(STOP_COMMAND_RE.search(command))

    def _queue_stop_waypoint(self) -> None:
        stop_wp = {
            "id": str(self._next_id),
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, # Dummy pose
            "description": "STOP",
            "priority": True,
            "mode": "nav2",
        }
        self._next_id += 1
        self._append_waypoints([stop_wp])

    def _queue_resume_waypoint(self) -> None:
        resume_wp = {
            "id": str(self._next_id),
            "pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}, # Dummy pose
            "description": "RESUME",
            "priority": True,
            "mode": "nav2",
        }
        self._next_id += 1
        self._append_waypoints([resume_wp])

    def _validate_step(self, step: dict) -> bool:
        action = str(step.get("action", "")).strip().lower()
        if action not in ALLOWED_ACTIONS:
            return False

        try:
            count = int(step.get("count", 1))
        except Exception:
            return False
        if count < 1 or count > 20:
            return False

        if action == "move":
            direction = str(step.get("direction", "")).strip().lower()
            if direction not in MOVE_DIRECTIONS:
                return False

            try:
                distance_m = float(step.get("distance_m", 0.0))
            except Exception:
                return False

            if distance_m <= 0.0:
                return False

            return True

        if action == "turn":
            direction = str(step.get("direction", "")).strip().lower()
            if direction not in TURN_DIRECTIONS:
                return False

            try:
                angle_deg = float(step.get("angle_deg", 0.0))
            except Exception:
                return False

            if angle_deg <= 0.0:
                return False

            return True

        if action == "goto":
            target_name = str(step.get("target_name", "")).strip()
            return bool(target_name)

        return False

    def _build_waypoints_from_steps(
        self,
        steps: list,
        pose: Optional[dict],
        command_mode_override: Optional[str],
        command_priority_override: Optional[bool],
    ) -> list:
        waypoints = []

        cur_x = float(pose["x"]) if pose else 0.0
        cur_y = float(pose["y"]) if pose else 0.0
        cur_yaw = float(pose["yaw"]) if pose else 0.0

        for step in steps:
            action = str(step.get("action", "")).strip().lower()
            count = int(step.get("count", 1))

            step_mode = self._parse_optional_mode(step.get("mode"))
            step_priority = self._parse_optional_priority(step.get("priority"))

            final_mode = command_mode_override or step_mode or self._default_mode_for_action(action)
            final_priority = (
                command_priority_override
                if command_priority_override is not None
                else (step_priority if step_priority is not None else False)
            )

            if action == "move":
                if pose is None:
                    self.get_logger().warn("Current pose unavailable; cannot compute relative move")
                    return []

                direction = str(step.get("direction", "")).strip().lower()
                distance_m = float(step.get("distance_m", 0.0))
                signed_distance = distance_m if direction == "forward" else -distance_m
                reverse_motion = direction == "backward"

                for i in range(count):
                    target_x = cur_x + signed_distance * math.cos(cur_yaw)
                    target_y = cur_y + signed_distance * math.sin(cur_yaw)

                    waypoints.append({
                        "id": str(self._next_id),
                        "pose": {
                            "x": round(target_x, 3),
                            "y": round(target_y, 3),
                            "yaw": round(cur_yaw, 4),
                        },
                        "description": f"move {direction} {distance_m:g}m ({i + 1}/{count})",
                        "priority": bool(final_priority),
                        "mode": final_mode,
                        "reverse": reverse_motion,
                    })
                    self._next_id += 1

                    cur_x, cur_y = target_x, target_y

                continue

            if action == "turn":
                if pose is None:
                    self.get_logger().warn("Current pose unavailable; cannot compute relative turn")
                    return []

                direction = str(step.get("direction", "")).strip().lower()
                angle_deg = float(step.get("angle_deg", 0.0))
                signed_angle = math.radians(angle_deg)
                if direction == "right":
                    signed_angle = -signed_angle

                for i in range(count):
                    target_yaw = self._normalize_angle(cur_yaw + signed_angle)

                    waypoints.append({
                        "id": str(self._next_id),
                        "pose": {
                            "x": round(cur_x, 3),
                            "y": round(cur_y, 3),
                            "yaw": round(target_yaw, 4),
                        },
                        "description": f"turn {direction} {angle_deg:g}deg ({i + 1}/{count})",
                        "priority": bool(final_priority),
                        "mode": final_mode,
                    })
                    self._next_id += 1

                    cur_yaw = target_yaw

                continue

            if action == "goto":
                target_name = str(step.get("target_name", "")).strip().lower()
                dest = self._resolve_named_destination(target_name)
                if dest is None:
                    self.get_logger().warn(f'Unknown named destination: "{target_name}"')
                    return []

                dest_mode = str(dest.get("mode", "nav2")).strip().lower()
                if dest_mode not in ALLOWED_MODES:
                    dest_mode = "nav2"

                final_goto_mode = command_mode_override or step_mode or dest_mode
                final_goto_priority = (
                    command_priority_override
                    if command_priority_override is not None
                    else (step_priority if step_priority is not None else bool(dest.get("priority", False)))
                )

                waypoints.append({
                    "id": str(self._next_id),
                    "pose": {
                        "x": float(dest["x"]),
                        "y": float(dest["y"]),
                        "yaw": float(dest.get("yaw", 0.0)),
                    },
                    "description": str(dest.get("description", f"goto {target_name}")),
                    "priority": bool(final_goto_priority),
                    "mode": final_goto_mode,
                })
                self._next_id += 1

                cur_x = float(dest["x"])
                cur_y = float(dest["y"])
                cur_yaw = float(dest.get("yaw", cur_yaw))
                continue

        return waypoints

    def _on_command(self, msg: String) -> None:
        command = msg.data.strip()
        if not command:
            return

        normalized_command = self._normalize_command_text(command)

        if self._processing:
            self._publish_status("BUSY: processing previous command, try again shortly")
            return

        self.get_logger().info(f'Mission command received: "{command}"')

        mode_override, priority_override = self._extract_command_overrides(command)

        is_stop = self._is_stop_command(normalized_command)
        is_resume = bool(RESUME_COMMAND_RE.search(normalized_command))

        if (
            self.robot_busy
            and not priority_override
            and not is_stop
            and not is_resume
        ):
            self.get_logger().warn(
                "Robot busy. Ignoring command."
            )

            self._publish_status("ROBOT_BUSY")
            return

        self._processing = True
        self._publish_status(f'PROCESSING: "{command}"')

        # Intercept emergency stops instantly
        if self._is_stop_command(normalized_command):
            try:
                self._queue_stop_waypoint()
                self._publish_status("STOP_REQUESTED")
            except Exception as e:
                self.get_logger().error(f"Stop command error: {e}")
            finally:
                self._processing = False
            return
        if RESUME_COMMAND_RE.search(normalized_command):
            try:
                self._queue_resume_waypoint()
                self._publish_status("RESUME_REQUESTED")
            except Exception as e:
                self.get_logger().error(f"Resume command error: {e}")
            finally:
                self._processing = False
            return
        
        self._executor_pool.submit(self._execute_wrapper, normalized_command)

    def _busy_callback(self, msg: Bool):
        self.robot_busy = msg.data


    def _execute_wrapper(self, command: str) -> None:
        try:
            self._execute(command)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"LLM returned invalid JSON: {e}")
            self._publish_status(f"ERROR: LLM returned unparseable response — {e}")
        except Exception as e:
            self.get_logger().error(f"Mission planner error: {e}")
            self._publish_status(f"ERROR: {e}")
        finally:
            self._processing = False

    def _execute(self, command: str) -> None:
        initial_pose = self._get_robot_pose()
        existing = self._load_waypoints()
        mode_override, priority_override = self._extract_command_overrides(command)

        history_text = "None"
        if self._chat_history:
            history_text = "\n".join([f"  {role.upper()}: {txt}" for role, txt in self._chat_history])

        user_msg = (
            f"ROBOT STATE:\n"
            f"  current_pose: {json.dumps(initial_pose) if initial_pose else 'unavailable'}\n"
            f"  next_id: {self._next_id}\n"
            f"  known_places: {json.dumps(self._named_destinations, indent=2)}\n"
            f"  queued_waypoints: {json.dumps(existing, indent=2) if existing else 'none'}\n"
            f"  command_mode_override: {mode_override if mode_override is not None else 'none'}\n"
            f"  command_priority_override: {priority_override if priority_override is not None else 'none'}\n"
            f"\nRECENT CONVERSATION HISTORY:\n{history_text}\n"
            f"\nOPERATOR COMMAND:\n\"{command}\"\n"
            f"\nReturn only the JSON structure requested in the system prompt."
        )

        t0 = time.time()
        raw = self._backend.complete(SYSTEM_PROMPT, user_msg, self._max_tokens)
        elapsed_ms = int((time.time() - t0) * 1000)

        self.get_logger().info(f"LLM responded in {elapsed_ms}ms")

        result = self._safe_parse_json(raw)
        understood = str(result.get("understood", "")).strip()
        steps = result.get("steps", [])
        notes = str(result.get("notes", "")).strip()

        self._chat_history.append(("operator", command))
        self._chat_history.append(("robot", understood))

        self.get_logger().info(f"Understood: {understood}")
        if notes:
            self.get_logger().warn(f"LLM notes: {notes}")

        if not isinstance(steps, list):
            self._publish_status("ERROR: LLM returned invalid steps format")
            return

        valid_steps = []
        for step in steps:
            if isinstance(step, dict) and self._validate_step(step):
                valid_steps.append(step)

        if not valid_steps:
            self._publish_status(f"NO_ACTION: {notes or understood}")
            return

        # Sample FRESH pose after LLM finishes
        fresh_pose = self._get_robot_pose()
        if fresh_pose is None and self._command_requires_pose(command):
            self._publish_status("ERROR: current pose unavailable for relative motion")
            return

        waypoints = self._build_waypoints_from_steps(
            valid_steps,
            fresh_pose,
            mode_override,
            priority_override,
        )

        if not waypoints:
            self._publish_status(f"NO_ACTION: {notes or understood}")
            return

        self._append_waypoints(waypoints)

        status = (
            f"DISPATCHED ({elapsed_ms}ms): {understood} | "
            f"{len(waypoints)} waypoint(s) queued"
        )
        if notes:
            status += f" | NOTE: {notes}"
        self._publish_status(status)

    def _cleanup_session_files(self) -> None:
        try:
            for tmp_file in self._waypoints_path.parent.glob(f".{self._waypoints_path.stem}.*.tmp"):
                try:
                    tmp_file.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        if not self._save_waypoints:
            try:
                if self._waypoints_file_existed:
                    self._waypoints_path.write_text(self._waypoints_backup_text)
                else:
                    if self._waypoints_path.exists():
                        self._waypoints_path.unlink()
            except Exception as e:
                self.get_logger().warn(f"Failed to restore waypoint file on shutdown: {e}")

    def destroy_node(self):
        if hasattr(self, "_executor_pool"):
            self._executor_pool.shutdown(wait=False)

        self._cleanup_session_files()
        self.get_logger().info("Shutting down LLMMissionPlanner")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LLMMissionPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()