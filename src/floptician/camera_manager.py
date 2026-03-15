import contextlib
import logging
import platform
import queue
import re
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from floptician.exceptions import CameraError

logger = logging.getLogger(__name__)


# --- Platform Helpers ---
def is_macos():
    return platform.system() == "Darwin"


def is_windows():
    return platform.system() == "Windows"


def _get_cap_prop_fps():
    fps_prop = getattr(cv2, "CAP_PROP_FPS", None)
    if fps_prop is not None:
        return fps_prop
    return getattr(cv2, "CAP_PROP_FRAME_FPS", None)


# --- FFmpeg Discovery ---
def _find_ffmpeg():
    exe = shutil.which("ffmpeg")
    return exe if exe else get_ffmpeg_exe()


# --- FFmpeg-based Camera Capture (macOS) ---
class FFmpegCameraCapture:
    """
    FFmpeg capture via AVFoundation on macOS.
    Tries low-latency and standard modes.
    """

    def __init__(self, device_index, width=1280, height=720, fps=30, buf_size=5):
        self.device_index = device_index
        self.width, self.height, self.fps = width, height, fps
        self.queue = queue.Queue(maxsize=buf_size)
        self.running = False
        self.proc = None
        self.thread = None
        self.ffmpeg = _find_ffmpeg()
        logger.info(f"Using FFmpeg: {self.ffmpeg}")

    def _build_commands(self):
        base = [
            self.ffmpeg,
            "-hide_banner",
            "-f",
            "avfoundation",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(self.fps),
            "-i",
            str(self.device_index),
        ]
        return [
            (
                "LowLatency",
                [
                    *base,
                    "-analyzeduration",
                    "0",
                    "-probesize",
                    "32",
                    "-fflags",
                    "nobuffer",
                    "-flags",
                    "low_delay",
                    "-pix_fmt",
                    "bgr24",
                    "-f",
                    "rawvideo",
                    "-",
                ],
            ),
            ("Standard", [*base, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]),
        ]

    def start(self):
        if self.running:
            return True
        for mode, cmd in self._build_commands():
            logger.debug(f"Trying FFmpeg mode {mode}: {' '.join(cmd)}")
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10 * 1024 * 1024
                )
                time.sleep(0.3)
                if self.proc.poll() is not None:
                    err = self.proc.stderr.read().decode(errors="ignore")
                    logger.debug(f"Mode {mode} failed: {err.strip()}")
                    continue
                self.running = True
                self.thread = threading.Thread(target=self._reader, daemon=True)
                self.thread.start()
                logger.info(f"FFmpeg capture started ({mode}) on device {self.device_index}")
                return True
            except Exception as e:
                logger.error(f"Start error ({mode}): {e}")
                if self.proc:
                    self.proc.terminate()
        raise CameraError("FFmpeg capture failed on all modes")

    def _reader(self):
        frame_size = self.width * self.height * 3
        errors = 0
        while self.running:
            data = self.proc.stdout.read(frame_size)
            if not data or len(data) < frame_size:
                errors += 1
                if errors >= 5:
                    break
                time.sleep(0.1)
                continue
            errors = 0
            frame = np.frombuffer(data, np.uint8).reshape(self.height, self.width, 3)
            if self.queue.full():
                with contextlib.suppress(queue.Empty):
                    self.queue.get_nowait()
            self.queue.put(frame)
        if self.proc and self.proc.poll() is not None:
            err = self.proc.stderr.read().decode(errors="ignore")
            logger.error(f"FFmpeg error: {err.strip()}")
        self.running = False
        logger.info("FFmpeg reader stopped")

    def read(self, timeout=1.0):
        if not self.running:
            return False, None
        try:
            return True, self.queue.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.running = False
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(0.5)
            except Exception:
                self.proc.kill()
        if self.thread and self.thread.is_alive():
            self.thread.join(0.5)
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break
        logger.info(f"FFmpeg capture stopped on device {self.device_index}")


# --- Camera Manager ---
try:
    from pygrabber.dshow_graph import FilterGraph

    windows_dshow = True
except ImportError:
    windows_dshow = False
    logger.warning("pygrabber not available - DirectShow disabled")


class CameraManager:
    """Unified interface: FFmpeg on macOS, DirectShow on Windows, OpenCV fallback."""

    def __init__(self):
        self.cameras = []
        self.cap = None
        self.ident = None
        self.width = None
        self.height = None
        self.fps = None
        self.refresh_camera_list()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_camera()
        return False

    def refresh_camera_list(self):
        self.cameras.clear()
        if is_macos():
            out = subprocess.run(
                [_find_ffmpeg(), "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True,
                text=True,
            ).stderr
            video = False
            for line in out.splitlines():
                if "AVFoundation video devices" in line:
                    video = True
                    continue
                if "AVFoundation audio devices" in line:
                    video = False
                    continue
                if video:
                    m = re.search(r"\[(\d+)\] (.*)", line)
                    if m:
                        self.cameras.append({"camera_index": int(m.group(1)), "camera_name": m.group(2)})
        elif is_windows() and windows_dshow:
            graph = FilterGraph()
            devs = graph.get_input_devices()
            self.cameras = [{"camera_index": i, "camera_name": n} for i, n in enumerate(devs)]
        else:
            for i in range(5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened() and cap.read()[0]:
                    self.cameras.append({"camera_index": i, "camera_name": f"Camera {i}"})
                cap.release()
        return self.cameras

    def get_available_cameras(self):
        return self.cameras

    def open_camera(self, ident, width=1280, height=720, fps=30):
        self.release_camera()
        self.ident, self.width, self.height, self.fps = ident, width, height, fps
        idx = ident["camera_index"] if isinstance(ident, dict) else int(ident)
        if is_macos():
            cap = FFmpegCameraCapture(idx, width, height, fps)
            ok = cap.start()
        else:
            flags = cv2.CAP_DSHOW if is_windows() and windows_dshow else 0
            cap = cv2.VideoCapture(idx, flags) if flags else cv2.VideoCapture(idx)
            ok = cap.isOpened()
            if ok:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                fps_prop = _get_cap_prop_fps()
                if fps_prop is not None:
                    cap.set(fps_prop, fps)
                else:
                    logger.warning("OpenCV FPS capture property is unavailable; skipping FPS configuration")
        self.cap = cap if ok else None
        return ok

    def set_resolution(self, width, height):
        return self.open_camera(self.ident, width, height, self.fps)

    def get_frame(self):
        return self.cap.read() if self.cap else (False, None)

    def release_camera(self):
        if not self.cap:
            return
        if hasattr(self.cap, "stop"):
            self.cap.stop()
        else:
            self.cap.release()
        self.cap = None

    def check_camera_status(self):
        if not self.cap:
            return False
        return getattr(self.cap, "running", False) if is_macos() else self.cap.grab()


# Utility
def list_all_cameras():
    return CameraManager().get_available_cameras()
