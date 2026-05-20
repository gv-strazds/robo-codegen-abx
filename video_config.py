"""Pure-python video / snapshot capture configuration.

This module is intentionally free of any Isaac Sim / USD / PhysX dependency so it
can be imported from mock tasks, tests, and CLI argument handling without
starting SimulationApp. The Isaac Sim-dependent recorder lives in
`video_capture.py`.

Two capture modes are anticipated:

* Streaming video (implemented in v1): one frame per render boundary, encoded
  directly to MP4 via ffmpeg. Low-resolution defaults so it stays cheap.
* Event-driven snapshots (future): higher-resolution PNGs at a much lower
  cadence, triggered by BT transitions ("item picked & lifted",
  "item placed at target", ...). Defaults are defined here so the schema is
  stable, but the implementation does not yet exist.

Both modes share `VideoCameraPreset` and the `VIDEO_CAMERA_PRESETS` registry.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple

from env_config_values import (
    CAMERA_VIEW1_LOOKAT,
    CAMERA_VIEW1_POS,
    CAMERA_VIEW1_WIDE_LOOKAT,
    CAMERA_VIEW1_WIDE_POS,
)


# --- Streaming-video defaults (v1) ----------------------------------------

DEFAULT_VIDEO_RESOLUTION: Tuple[int, int] = (1280, 720)
DEFAULT_VIDEO_FPS: float = 30.0  # informational; actual fps = 1/(physics_dt*psteps_per_render)
DEFAULT_VIDEO_CODEC: str = "libx264"
DEFAULT_VIDEO_QUALITY: int = 8  # imageio scale, 0..10 (higher = better, larger file)
DEFAULT_VIDEO_OUTPUT_DIR: str = "_results/videos"


# --- Snapshot defaults (future feature; placeholders for schema stability) ---

DEFAULT_SNAPSHOT_RESOLUTION: Tuple[int, int] = (1920, 1080)
DEFAULT_SNAPSHOT_FPS: float = 3.0
DEFAULT_SNAPSHOT_OUTPUT_DIR: str = "_results/snapshots"
DEFAULT_SNAPSHOT_FORMAT: str = "png"
# Render boundaries to wait between an event-trigger and the actual capture,
# so the scene has a chance to visually settle (e.g. lift completes, gripper
# fully closes) before the frame is grabbed.
DEFAULT_SNAPSHOT_SETTLING_FRAMES: int = 3


# --- Camera presets -------------------------------------------------------

@dataclass(frozen=True)
class VideoCameraPreset:
    name: str
    eye: Tuple[float, float, float]
    target: Tuple[float, float, float]
    focal_length: float = 24.0  # mm
    horizontal_aperture: float = 20.955  # mm (matches Isaac Sim default)
    clipping_range: Tuple[float, float] = (0.05, 1000.0)


def _to_tuple3(arr) -> Tuple[float, float, float]:
    return (float(arr[0]), float(arr[1]), float(arr[2]))


VIDEO_CAMERA_PRESETS: Dict[str, VideoCameraPreset] = {
    "view1": VideoCameraPreset(
        name="view1",
        eye=_to_tuple3(CAMERA_VIEW1_POS),
        target=_to_tuple3(CAMERA_VIEW1_LOOKAT),
    ),
    # Initial copy of view1 — separate registry entry so snapshot framing
    # can be retuned without touching the streaming-video preset.
    "snapshot1": VideoCameraPreset(
        name="snapshot1",
        eye=_to_tuple3(CAMERA_VIEW1_POS),
        target=_to_tuple3(CAMERA_VIEW1_LOOKAT),
    ),
    # Pulled back further along the same view vector for streaming video
    # and wide-shot snapshots (every-N cadence + final task_verified frame).
    "video1": VideoCameraPreset(
        name="video1",
        eye=_to_tuple3(CAMERA_VIEW1_WIDE_POS),
        target=_to_tuple3(CAMERA_VIEW1_WIDE_LOOKAT),
    ),
}


def make_default_video_path(
    task_name: str,
    output_dir: str = DEFAULT_VIDEO_OUTPUT_DIR,
    ext: str = "mp4",
) -> str:
    """Build a timestamped output path: {output_dir}/{task_name}_{YYYYmmdd_HHMMSS}.{ext}."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{output_dir.rstrip('/')}/{task_name}_{stamp}.{ext}"


def make_default_snapshot_dir(
    task_name: str,
    output_dir: str = DEFAULT_SNAPSHOT_OUTPUT_DIR,
) -> str:
    """Build a per-run snapshot directory path: {output_dir}/{task_name}_{YYYYmmdd_HHMMSS}/.

    One-second timestamp resolution matches make_default_video_path.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{output_dir.rstrip('/')}/{task_name}_{stamp}"


def parse_resolution(spec: str) -> Tuple[int, int]:
    """Parse a 'WxH' string (e.g. '960x540') into a (width, height) tuple of ints.

    Raises ValueError on malformed input.
    """
    s = spec.strip().lower().replace("X", "x")
    parts = s.split("x")
    if len(parts) != 2:
        raise ValueError(f"resolution must be 'WxH', got {spec!r}")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError(f"resolution width and height must be positive, got {spec!r}")
    return (w, h)
