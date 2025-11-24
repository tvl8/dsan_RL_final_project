#!/usr/bin/env python3
"""
QuakeSingleMapEnv: Gymnasium-style environment for a single-player Quake map.

Phase C.1: Skeleton only
- Defines class structure
- Handles engine process lifecycle (launch/close)
- Stubs for observation, reward, and step logic

Later phases (C.2+):
- Wire in real game-state extraction from FTEQW
- Implement observation vector from engine state
- Implement reward function and termination logic

Phase C.2:
- Adds real Quake engine input control
- Implements action→command mapping
- Implements press/hold/release logic with frame_skip
- Leaves observation + reward as placeholders for C.3/C.4

Phase C.3 prototype:
- Real Quake input (movement/turn/strafe/fire/jump) via FTE commands.
- Prototype game state parsing from engine stdout (position, angles, health, ammo, kills).
- Prototype R1 reward:
    +10 per kill
    +100 per progress zone (stubbed as a float field for now)
    -2 per point of damage taken
    -0.01 * frame_skip per step (time penalty)
    -200 on death

IMPORTANT:
- The regex patterns in `_handle_engine_line` are guesses and WILL need tuning
  based on the actual FTE console output on your machine.
- Enable the debug print in `_handle_engine_line` to see raw lines and adjust.
"""

from __future__ import annotations

import subprocess
import sys
import time
import re
import select
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


# ---------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------

@dataclass
class QuakeEnvConfig:
    """
    Configuration for QuakeSingleMapEnv.

    You can tune:
    - frame_skip: how long each action is held.
    - poll_timeout / poll_iterations: how aggressively we read stdout.
    """
    map_name: str = "e1m1"
    frame_skip: int = 4
    max_episode_steps: int = 5000
    windowed: bool = True
    extra_args: str = ""
    poll_timeout: float = 0.02
    poll_iterations: int = 5


# ---------------------------------------------------------
# Utility: repo root + engine paths
# ---------------------------------------------------------

def find_repo_root() -> Path:
    """
    Walk upward from this file until we see README.md; assume that is repo root.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "README.md").exists():
            return parent
    # Fallback: assume repo_root/code/quake/envs/this_file.py
    return Path(__file__).resolve().parents[3]


def build_fteqw_paths(repo_root: Path) -> Dict[str, Path]:
    """
    Return paths for FTEQW Windows build and Quake assets.
    """
    fte_base = repo_root / "code" / "quake" / "fteqw" / "windows"
    exe_path = fte_base / "fteqw64.exe"
    id1_dir = fte_base / "id1"
    pak0 = id1_dir / "pak0.pak"
    pak1 = id1_dir / "pak1.pak"
    return {
        "fte_base": fte_base,
        "exe_path": exe_path,
        "id1_dir": id1_dir,
        "pak0": pak0,
        "pak1": pak1,
    }


def verify_fte_install(paths: Dict[str, Path]) -> None:
    """
    Ensure engine executable and pak files exist.
    """
    missing = []
    if not paths["exe_path"].exists():
        missing.append(f"Missing executable: {paths['exe_path']}")
    if not paths["pak0"].exists():
        missing.append(f"Missing pak0.pak: {paths['pak0']}")
    if not paths["pak1"].exists():
        missing.append(f"Missing pak1.pak: {paths['pak1']}")
    if missing:
        for m in missing:
            print(m, file=sys.stderr)
        raise RuntimeError("FTEQW install incomplete.")


# ---------------------------------------------------------
# Environment Class
# ---------------------------------------------------------

class QuakeSingleMapEnv(gym.Env):
    """
    Gymnasium-compatible environment for a single Quake map (e.g., e1m1).

    Observations (prototype):
        16-d float vector:
        [x, y, z, yaw, pitch,
         health, armor,
         ammo_shells, ammo_nails, ammo_rockets, ammo_cells,
         nearest_enemy_dist, nearest_enemy_angle,
         progress_zone, time_since_start]

    Actions (discrete macro-actions):
        0: no-op
        1: move forward
        2: move backward
        3: turn left
        4: turn right
        5: strafe left
        6: strafe right
        7: fire
        8: jump
    """

    metadata = {"render_modes": ["human", "none"], "render_fps": 35}

    def __init__(self, config: Optional[QuakeEnvConfig] = None, render_mode: str = "human"):
        super().__init__()

        self.config = config or QuakeEnvConfig()
        self.render_mode = render_mode

        # Paths
        self.repo_root: Path = find_repo_root()
        self.fte_paths: Dict[str, Path] = build_fteqw_paths(self.repo_root)
        verify_fte_install(self.fte_paths)

        # Engine process
        self.engine_proc: Optional[subprocess.Popen] = None

        # Episode bookkeeping
        self.current_step: int = 0
        self.episode_start_time: float = time.time()

        # Internal numeric state (proto)
        self._state: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "health": 100.0,
            "armor": 0.0,
            "ammo_shells": 0.0,
            "ammo_nails": 0.0,
            "ammo_rockets": 0.0,
            "ammo_cells": 0.0,
            "nearest_enemy_dist": 0.0,
            "nearest_enemy_angle": 0.0,
            "progress_zone": 0.0,  # will be computed from position later
            "time_since_start": 0.0,
        }
        self._last_state: Dict[str, float] = self._state.copy()

        # Kill tracking for reward
        self._kills: int = 0
        self._last_kills: int = 0

        # Action/observation spaces
        self.action_space = Discrete(9)
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32,
        )

        # Launch engine
        self._launch_engine()

    # -----------------------------------------------------
    # Engine lifecycle
    # -----------------------------------------------------

    def _build_engine_command(self) -> list[str]:
        exe = str(self.fte_paths["exe_path"])
        cmd: list[str] = [exe]

        if self.config.windowed:
            cmd.extend(["+set", "vid_fullscreen", "0"])

        if self.config.map_name:
            cmd.extend(["+map", self.config.map_name])

        if self.config.extra_args:
            cmd.extend(self.config.extra_args.split())

        return cmd

    def _launch_engine(self) -> None:
        if self.engine_proc is not None and self.engine_proc.poll() is None:
            return

        cmd = self._build_engine_command()
        print("[QuakeEnv] Launching engine:", " ".join(cmd))

        try:
            self.engine_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except Exception as e:
            print(f"[QuakeEnv] Engine launch failed: {e}", file=sys.stderr)
            raise

    def _shutdown_engine(self) -> None:
        if self.engine_proc is None:
            return
        try:
            self.engine_proc.terminate()
        except Exception as e:
            print(f"[QuakeEnv] Engine termination failed: {e}", file=sys.stderr)
        self.engine_proc = None

    # -----------------------------------------------------
    # Engine I/O helpers
    # -----------------------------------------------------

    def _send_cmd(self, cmd: str) -> None:
        """
        Send a single console command to FTE.
        """
        if self.engine_proc is None or self.engine_proc.stdin is None:
            return
        try:
            self.engine_proc.stdin.write((cmd + "\n").encode())
            self.engine_proc.stdin.flush()
        except Exception as e:
            print(f"[QuakeEnv] Error sending command '{cmd}': {e}", file=sys.stderr)

    def _press_and_release(self, key: str, ticks: int) -> None:
        """
        Hold +key for `ticks` Quake frames (approx), then release -key.
        """
        self._send_cmd(f"+{key}")
        for _ in range(ticks):
            time.sleep(1 / 35.0)  # ~35 FPS
            self._poll_engine_output(self.config.poll_timeout)
        self._send_cmd(f"-{key}")

    def _poll_engine_output(self, timeout: float) -> None:
        """
        Non-blocking-ish read from engine stdout with select().

        For each readable line, call _handle_engine_line(line).
        """
        if self.engine_proc is None or self.engine_proc.stdout is None:
            return

        stdout = self.engine_proc.stdout

        for _ in range(self.config.poll_iterations):
            try:
                rlist, _, _ = select.select([stdout], [], [], timeout)
            except Exception:
                break

            if not rlist:
                break

            try:
                raw = stdout.readline()
            except Exception:
                break

            if not raw:
                break

            try:
                line = raw.decode(errors="ignore").strip()
            except Exception:
                continue

            if line:
                self._handle_engine_line(line)

    def _handle_engine_line(self, line: str) -> None:
        """
        Parse a single console line from FTE and update internal state.

        IMPORTANT: The regex patterns here are guesses.
        You should enable the debug print, observe real lines,
        and adjust patterns to match FTE's actual output.
        """
        # Uncomment this while debugging to see actual engine lines:
        print("[FTE]", line)

        # Example guessed output for a position/angle query like 'viewpos'
        # e.g., "viewpos: 100.0 50.0 30.0 90.0 0.0"
        m = re.search(
            r"viewpos:?\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)",
            line,
        )
        if m:
            self._state["x"] = float(m.group(1))
            self._state["y"] = float(m.group(2))
            self._state["z"] = float(m.group(3))
            self._state["yaw"] = float(m.group(4))
            self._state["pitch"] = float(m.group(5))
            return

        # Example guessed status line:
        # "health: 100 armor: 0 shells: 50 nails: 0 rockets: 0 cells: 0"
        m = re.search(
            r"health[: ]+([0-9]+)\s+armor[: ]+([0-9]+)\s+shells[: ]+([0-9]+)\s+nails[: ]+([0-9]+)\s+rockets[: ]+([0-9]+)\s+cells[: ]+([0-9]+)",
            line,
            re.IGNORECASE,
        )
        if m:
            self._state["health"] = float(m.group(1))
            self._state["armor"] = float(m.group(2))
            self._state["ammo_shells"] = float(m.group(3))
            self._state["ammo_nails"] = float(m.group(4))
            self._state["ammo_rockets"] = float(m.group(5))
            self._state["ammo_cells"] = float(m.group(6))
            return

        # Example guessed kill counter line: "kills: 3"
        m = re.search(r"kills[: ]+([0-9]+)", line, re.IGNORECASE)
        if m:
            self._kills = int(m.group(1))
            return

        # TODO: later, parse enemy positions, progress triggers, etc.

    # -----------------------------------------------------
    # Action mapping
    # -----------------------------------------------------

    def _apply_action(self, action: int) -> None:
        """
        Map discrete action index to Quake movement commands.
        """
        fs = self.config.frame_skip

        # 0: no-op
        if action == 0:
            for _ in range(fs):
                time.sleep(1 / 35.0)
                self._poll_engine_output(self.config.poll_timeout)
            return

        # 1: move forward
        if action == 1:
            self._press_and_release("forward", fs)
            return

        # 2: move backward
        if action == 2:
            self._press_and_release("back", fs)
            return

        # 3: turn left
        if action == 3:
            self._press_and_release("left", fs)
            return

        # 4: turn right
        if action == 4:
            self._press_and_release("right", fs)
            return

        # 5: strafe left
        if action == 5:
            self._press_and_release("moveleft", fs)
            return

        # 6: strafe right
        if action == 6:
            self._press_and_release("moveright", fs)
            return

        # 7: fire
        if action == 7:
            self._press_and_release("attack", fs)
            return

        # 8: jump
        if action == 8:
            self._press_and_release("jump", fs)
            return

        raise ValueError(f"Invalid action index: {action}")

    # -----------------------------------------------------
    # Observation + reward
    # -----------------------------------------------------

    def _build_observation(self) -> np.ndarray:
        """
        Build the observation vector from self._state.
        """
        self._state["time_since_start"] = float(time.time() - self.episode_start_time)

        ordered_keys = [
            "x",
            "y",
            "z",
            "yaw",
            "pitch",
            "health",
            "armor",
            "ammo_shells",
            "ammo_nails",
            "ammo_rockets",
            "ammo_cells",
            "nearest_enemy_dist",
            "nearest_enemy_angle",
            "progress_zone",
            "time_since_start",
        ]

        obs = np.array([self._state[k] for k in ordered_keys], dtype=np.float32)
        return obs

    def _compute_reward_and_termination(self) -> Tuple[float, bool]:
        """
        Implement R1 balanced reward prototype.

        +10 per kill
        +100 per progress zone increment (stubbed as a float)
        -2 per damage point
        -0.01 * frame_skip per step (time penalty)
        -200 if health <= 0 (death)
        """
        reward = 0.0
        terminated = False

        # Time penalty
        reward -= 0.01 * self.config.frame_skip

        # Damage penalty
        damage = max(0.0, self._last_state["health"] - self._state["health"])
        if damage > 0:
            reward -= 2.0 * damage

        # Kill reward
        kill_delta = self._kills - self._last_kills
        if kill_delta > 0:
            reward += 10.0 * kill_delta

        # Progress zone reward (to be wired later)
        progress_delta = self._state["progress_zone"] - self._last_state["progress_zone"]
        if progress_delta > 0:
            reward += 100.0 * progress_delta

        # Death penalty
        if self._state["health"] <= 0.0:
            reward -= 200.0
            terminated = True

        return reward, terminated

    # -----------------------------------------------------
    # Gym API
    # -----------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset Quake episode by reloading the map.
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.episode_start_time = time.time()

        # Reset internal state
        self._kills = 0
        self._last_kills = 0

        for k in self._state:
            if k == "health":
                self._state[k] = 100.0
            else:
                self._state[k] = 0.0
        self._last_state = self._state.copy()

        # Ensure engine is running and on correct map
        if self.engine_proc is None or self.engine_proc.poll() is not None:
            self._launch_engine()
        else:
            # Reload the map
            self._send_cmd(f"map {self.config.map_name}")

        # Give Quake a bit of time to load map + initial state
        time.sleep(1.0)
        self._poll_engine_output(self.config.poll_timeout)

        obs = self._build_observation()
        return obs, {}

    def step(self, action: int):
        """
        Apply action, advance engine, read new state, and compute reward.
        """
        assert self.action_space.contains(action)

        self.current_step += 1

        # Snapshot last state for reward calculation
        self._last_state = self._state.copy()
        self._last_kills = self._kills

        # Apply high-level action → Quake commands
        self._apply_action(action)

        # Poll engine output to update internal state
        self._poll_engine_output(self.config.poll_timeout)

        # Build observation
        obs = self._build_observation()

        # Compute R1 reward + termination
        reward, terminated = self._compute_reward_and_termination()

        # Truncation if we exceed max steps
        truncated = self.current_step >= self.config.max_episode_steps
        info: Dict[str, Any] = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        """
        Rendering is handled by FTEQW itself (actual game window).
        """
        return

    def close(self):
        """
        Close environment and terminate engine.
        """
        self._shutdown_engine()
        super().close()

    def __del__(self):
        try:
            self._shutdown_engine()
        except Exception:
            pass