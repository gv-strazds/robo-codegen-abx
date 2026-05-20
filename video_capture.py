"""Streaming MP4 recorder + still-image snapshot capture for IsaacSim runs.

`VideoRecorder` wraps an `isaacsim.sensors.camera.Camera` and pipes RGB
frames straight into a ffmpeg subprocess (libx264, yuv420p) via stdin.
One frame per call to `capture()`, intended to be invoked once per render
boundary in the main simulation loop.  ffmpeg is invoked from PATH so we
don't depend on the imageio backend that Isaac Sim happens to bundle (the
Kit-bundled imageio has no ffmpeg plugin and routes .mp4 to TiffWriter).

`SnapshotCapture` owns its own camera prim and writes individual PNGs:
either event-triggered (BT phase transitions, queued with a settling
delay) or time-based (every `time_period_s` of sim-time).  PNGs go
through `imageio.imwrite`, which has a built-in PNG plugin (no ffmpeg /
TiffWriter pitfall — PNG isn't routed through ffmpeg).  Each PNG is
accompanied by a sidecar JSON with the same basename + ".json".

Camera orientation is computed in pure numpy (look-at math producing a
quaternion in Camera's "world" axis convention: +X forward, +Z up). This
avoids `isaacsim.core.utils.viewports.set_camera_view`, which requires an
active GUI viewport and silently no-ops in headless mode.
"""

import json
import logging
import os
import shutil
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from video_config import (
    DEFAULT_SNAPSHOT_FPS,
    DEFAULT_SNAPSHOT_SETTLING_FRAMES,
    VideoCameraPreset,
)

logger = logging.getLogger(__name__)


def _lookat_quat_wxyz(
    eye: Tuple[float, float, float],
    target: Tuple[float, float, float],
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return a wxyz quaternion that rotates the Camera's local frame so its
    +X axis points from `eye` toward `target` and its +Z axis is as close to
    world `up` as possible.

    Matches Isaac Sim Camera.set_world_pose(camera_axes='world'), where the
    'world' axis convention is +X forward, +Z up.
    """
    e = np.asarray(eye, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)

    forward = t - e
    fnorm = np.linalg.norm(forward)
    if fnorm < 1e-8:
        raise ValueError(f"eye and target are coincident: {eye} -> {target}")
    forward /= fnorm

    z_axis = u - forward * np.dot(u, forward)
    if np.linalg.norm(z_axis) < 1e-6:
        u = np.array([0.0, 1.0, 0.0])
        z_axis = u - forward * np.dot(u, forward)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, forward)
    y_axis /= np.linalg.norm(y_axis)

    R = np.column_stack([forward, y_axis, z_axis])
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _normalize_rgb(rgb, expected_resolution: Tuple[int, int]) -> Optional[np.ndarray]:
    """Coerce a Camera.get_rgb() return value to a uint8 RGB ndarray of the
    expected (W, H) resolution.  Returns None if the frame is empty, the
    wrong shape, or otherwise unusable.
    """
    if rgb is None:
        return None
    if not isinstance(rgb, np.ndarray):
        try:
            rgb = rgb.numpy()
        except Exception:
            rgb = np.asarray(rgb)
    if rgb.size == 0:
        return None
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    expected_w, expected_h = expected_resolution
    if rgb.shape[0] != expected_h or rgb.shape[1] != expected_w:
        return None
    if not rgb.flags["C_CONTIGUOUS"]:
        rgb = np.ascontiguousarray(rgb)
    return rgb


def _make_camera_from_preset(
    preset: VideoCameraPreset,
    prim_path: str,
    name_prefix: str,
    resolution: Tuple[int, int],
):
    """Create an `isaacsim.sensors.camera.Camera` at `prim_path`, position it
    via `preset` (eye/target → look-at quaternion), and apply lens config.

    Lens config (focal length, aperture, clipping range) is best-effort: any
    failure logs a warning and falls back to camera defaults.
    """
    from isaacsim.sensors.camera import Camera

    camera = Camera(
        prim_path=prim_path,
        name=f"{name_prefix}_{preset.name}",
        resolution=resolution,
    )
    quat_wxyz = _lookat_quat_wxyz(preset.eye, preset.target)
    camera.set_world_pose(
        position=np.asarray(preset.eye, dtype=np.float64),
        orientation=quat_wxyz,
        camera_axes="world",
    )
    try:
        camera.set_focal_length(preset.focal_length)
        camera.set_horizontal_aperture(preset.horizontal_aperture)
        camera.set_clipping_range(
            near_distance=preset.clipping_range[0],
            far_distance=preset.clipping_range[1],
        )
    except Exception as e:
        logger.warning(
            f"{name_prefix}: optional lens config failed ({e}); using camera defaults."
        )
    return camera


class VideoRecorder:
    def __init__(
        self,
        preset: VideoCameraPreset,
        output_path: str,
        resolution: Tuple[int, int],
        fps: float,
        prim_path: str = "/World/video_camera",
    ) -> None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self._preset = preset
        self._output_path = output_path
        self._resolution = resolution
        self._fps = float(fps)
        self._prim_path = prim_path
        self._frame_count = 0
        self._writer = None
        self._closed = False

        self._camera = _make_camera_from_preset(
            preset, prim_path, "VideoRecorder", resolution
        )

    def initialize(self) -> None:
        """Attach the RGB annotator and spawn the ffmpeg encoder subprocess.

        Must be called *after* `world.reset()`, otherwise the underlying render
        product can't be created.
        """
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError(
                "ffmpeg not found on PATH; install ffmpeg or add it to PATH "
                "before enabling video capture."
            )

        self._camera.initialize()

        w, h = self._resolution
        cmd = [
            ffmpeg_bin, "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", f"{self._fps:.6f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            "-preset", "fast",
            self._output_path,
        ]
        self._writer = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.warning(
            f"VideoRecorder: opened {self._output_path} "
            f"({w}x{h} @ {self._fps:.2f} fps, camera={self._preset.name})"
        )

    def capture(self) -> bool:
        """Grab one RGB frame and feed it to ffmpeg. Returns True on success."""
        if self._writer is None or self._closed or self._writer.stdin is None:
            return False
        rgb = _normalize_rgb(self._camera.get_rgb(), self._resolution)
        if rgb is None:
            return False
        try:
            self._writer.stdin.write(rgb.tobytes())
        except BrokenPipeError:
            stderr = self._writer.stderr.read() if self._writer.stderr else b""
            logger.warning(
                f"VideoRecorder: ffmpeg pipe broken after {self._frame_count} frames. "
                f"stderr: {stderr.decode(errors='replace')}"
            )
            self._closed = True
            return False
        self._frame_count += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            try:
                if self._writer.stdin is not None:
                    self._writer.stdin.close()
                rc = self._writer.wait(timeout=30)
                if rc != 0:
                    stderr = self._writer.stderr.read() if self._writer.stderr else b""
                    logger.warning(
                        f"VideoRecorder: ffmpeg exited rc={rc}. stderr: {stderr.decode(errors='replace')}"
                    )
            except Exception as e:
                logger.warning(f"VideoRecorder: ffmpeg shutdown failed: {e}")
                try:
                    self._writer.kill()
                except Exception:
                    pass
            self._writer = None
        logger.warning(
            f"VideoRecorder: wrote {self._frame_count} frames to {self._output_path}"
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def output_path(self) -> str:
        return self._output_path


class SnapshotCapture:
    """Event-triggered + time-based PNG capture from a dedicated camera.

    Lifecycle (mirrors VideoRecorder): __init__ → initialize → tick* + request_event* → close.

    Two trigger sources, both writing into the same per-run directory:
      * Event: `request_event(name, sim_time, pick_index)` queues a capture
        with `settling_frames` render-boundary delay so the scene visually
        settles after the BT phase fires.  Filename: `{name}_pick{N:02}_t{sim_time:.3f}.png`.
      * Time: every `time_period_s` of sim-time, `tick()` saves a sequential
        `0001.png`, `0002.png`, ... .

    Each PNG is paired with a sidecar `.json` containing the trigger info plus
    optional `pick_name` / `target_name` (whatever `metadata_provider` returns).
    """

    def __init__(
        self,
        preset: VideoCameraPreset,
        output_dir: str,
        resolution: Tuple[int, int],
        settling_frames: int = DEFAULT_SNAPSHOT_SETTLING_FRAMES,
        time_period_s: float = 1.0 / DEFAULT_SNAPSHOT_FPS,
        prim_path: str = "/World/snapshot_camera",
        metadata_provider: Optional[Callable[[], Dict]] = None,
        time_based_enabled: bool = True,
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        self._preset = preset
        self._output_dir = output_dir
        self._resolution = resolution
        self._settling_frames = int(settling_frames)
        self._time_period_s = float(time_period_s)
        self._prim_path = prim_path
        self._metadata_provider = metadata_provider or (lambda: {})
        self._time_based_enabled = bool(time_based_enabled)

        # (remaining_frames, basename_without_ext, sidecar_metadata)
        self._pending_event_captures: List[Tuple[int, str, Dict]] = []
        self._time_based_count: int = 0
        self._last_time_based_sim_t: Optional[float] = None
        self._total: int = 0
        self._closed = False

        self._camera = _make_camera_from_preset(
            preset, prim_path, "SnapshotCapture", resolution
        )

    def initialize(self) -> None:
        """Attach the RGB annotator. Must be called *after* `world.reset()`."""
        self._camera.initialize()
        logger.warning(
            f"SnapshotCapture: opened {self._output_dir} "
            f"({self._resolution[0]}x{self._resolution[1]}, settling={self._settling_frames}, "
            f"period={self._time_period_s:.3f}s, camera={self._preset.name})"
        )

    def request_event(
        self,
        event_name: str,
        sim_time: float,
        pick_index: int,
        extra_metadata: Optional[Dict] = None,
    ) -> None:
        """Queue an event-triggered capture; fires after `settling_frames` ticks.

        ``extra_metadata`` is merged into the JSON sidecar **after** the
        ``metadata_provider`` callback, so caller-supplied keys win over the
        provider's defaults.  Useful when the event is about a *specific*
        pick/target that may differ from the live ``TaskContext`` state at
        request time (e.g. an incremental verification failure references
        a previously-completed pick, while the context is already tracking
        the next one).
        """
        if self._closed:
            return
        basename = f"{event_name}_pick{pick_index:02}_t{sim_time:.3f}"
        meta = {
            "kind": "event",
            "event": event_name,
            "sim_time": float(sim_time),
            "pick_index": int(pick_index),
        }
        try:
            meta.update(self._metadata_provider() or {})
        except Exception as e:
            logger.warning(f"SnapshotCapture: metadata_provider raised: {e}")
        if extra_metadata:
            meta.update(extra_metadata)
        self._pending_event_captures.append((self._settling_frames, basename, meta))

    def tick(self, sim_time: float, pick_index: int) -> None:
        """Drive pending event captures and the time-based cadence.

        Should be called once per render boundary while the world is playing.
        """
        if self._closed:
            return

        # Event-driven: decrement settling counters; capture those at 0.
        if self._pending_event_captures:
            still_pending: List[Tuple[int, str, Dict]] = []
            for remaining, basename, meta in self._pending_event_captures:
                if remaining <= 0:
                    self._capture_to(basename, meta)
                else:
                    still_pending.append((remaining - 1, basename, meta))
            self._pending_event_captures = still_pending

        # Time-based: fire on the first tick, then every `time_period_s` of sim-time.
        # Skipped entirely in failure-only mode (--snapshot-errors), where the
        # constant cadence captures would defeat the purpose of the mode.
        if not self._time_based_enabled:
            return
        if self._last_time_based_sim_t is None or (
            sim_time - self._last_time_based_sim_t >= self._time_period_s
        ):
            self._last_time_based_sim_t = sim_time
            self._time_based_count += 1
            basename = f"{self._time_based_count:04d}"
            meta = {
                "kind": "time_based",
                "sim_time": float(sim_time),
                "pick_index": int(pick_index),
                "sequence": self._time_based_count,
            }
            try:
                meta.update(self._metadata_provider() or {})
            except Exception as e:
                logger.warning(f"SnapshotCapture: metadata_provider raised: {e}")
            self._capture_to(basename, meta)

    def _capture_to(self, basename: str, metadata: Dict) -> None:
        rgb = _normalize_rgb(self._camera.get_rgb(), self._resolution)
        if rgb is None:
            logger.warning(
                f"SnapshotCapture: dropping {basename} (camera returned empty/wrong-shape frame)"
            )
            return
        png_path = os.path.join(self._output_dir, f"{basename}.png")
        json_path = os.path.join(self._output_dir, f"{basename}.json")
        try:
            import imageio
            imageio.imwrite(png_path, rgb)
        except Exception as e:
            logger.warning(f"SnapshotCapture: imwrite failed for {png_path}: {e}")
            return
        sidecar = dict(metadata)
        sidecar["filename"] = f"{basename}.png"
        try:
            with open(json_path, "w") as f:
                json.dump(sidecar, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"SnapshotCapture: sidecar write failed for {json_path}: {e}")
        self._total += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Flush any remaining queued captures (e.g. a task_finished event that
        # didn't reach its settling delay before the loop exited).
        for remaining, basename, meta in self._pending_event_captures:
            self._capture_to(basename, meta)
        self._pending_event_captures = []
        logger.warning(
            f"SnapshotCapture: wrote {self._total} snapshots to {self._output_dir}"
        )

    @property
    def total(self) -> int:
        return self._total

    @property
    def output_dir(self) -> str:
        return self._output_dir
