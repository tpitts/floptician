Floptician
==========

Floptician is a poker table card-detection project built around live capture, YOLO-based recognition, and OBS/browser overlay output.

Run
---

Follow [SETUP.md](SETUP.md) for Windows and Apple Silicon Mac setup. Windows users
can use `setup.bat` and `run.bat`. After syncing the environment, run:

```bash
uv run --no-sync floptician run
```

Runtime Notes
-------------

- Copy `config.example.yaml` to `config.yaml` for machine-local settings.
- Local runtime configuration is read from `config.yaml` by default.
- You can also point the app at another config file with `uv run --no-sync floptician run --config path/to/config.yaml` or `FLOPTICIAN_CONFIG=...`.
- `config.yaml` is intentionally ignored and is expected to stay local to each machine.
- Model paths in config point at files under `models/`.
- `output/` is used for generated run artifacts.
- If you use OBS capture mode, OBS with its WebSocket server must be available locally.

Board stability uses both consecutive readings and elapsed time. With the default
settings, initial boards, additions, corrections, and layout/position changes need
3 matching readings spanning at least 1.2 seconds (about 1.7 seconds at 1.8 FPS).
Cards, positions, and layout are confirmed together. A different candidate or an
unusable frame restarts confirmation; confidence changes alone do not.

Partial disappearance holds the whole board until confirmed cards remain missing
for 6 readings spanning at least 4.2 seconds (about 4.4 seconds at 1.8 FPS).
Seeing all confirmed cards again cancels clearing. Inference/capture failures do
not count as empty-table readings or refresh the overlay; the existing browser
inactivity timeout clears it after roughly 10 seconds without usable updates.

See [development and validation](SETUP.md#development-and-validation) for the
backend-specific sync commands. Run the full regression checks with
`uv run --no-sync python -m pytest -q --require-all-tests`. This requires Chrome,
the browser-test extra, and the tracked model weights. The flag prevents skipped
tests from being mistaken for a complete validation run. Browser checks use
isolated sessions and mocked WebSocket traffic, without connecting to OBS.

Dependencies
------------

uv manages `.venv` using the Python version in `.python-version` and exact
dependency versions in `uv.lock`. `pyproject.toml` defines the dependencies;
the lockfile is generated. On Windows, setup selects and remembers CPU or CUDA.
For Apple Silicon Mac:

```bash
uv sync --locked --no-dev
```

- Windows development syncs use `--extra windows` plus either `--extra cpu` or `--extra cuda`.
- Camera capture uses OpenCV: AVFoundation on macOS and DirectShow on Windows.
  Device discovery uses persistent IDs and the same ordering as the selected backend.
  macOS discovery dependencies (PyObjC) are installed automatically by uv.
- On Apple Silicon, add `--extra coreml`, run `scripts/convert.py`, then run
  `scripts/verify_coreml.py` before selecting the generated `.mlpackage` in `config.yaml`.
  The default `cpu-and-ne` Core ML policy leaves the GPU available to OBS.

Camera capture
--------------

Use `floptician list-cameras` to inspect connected devices. The interactive run
command offers an arrow-key camera menu. Device identity is resolved again before
opening, so a changed camera index cannot silently select a different device.

`capture.camera_width`, `camera_height`, and `camera_fps` request the native camera
mode (defaults: 1280×720 at 30 fps). Direct capture preserves native resolution;
`capture.width` and `height` apply only to OBS screenshots. `capture.fps` controls
detection sampling (default 1.8).
Delivered dimensions and the driver's reported FPS are logged; unsupported modes
produce a warning. The detector prepares its input once, preserving aspect ratio
with black padding when necessary; there is no intermediate downsize/upscale.
Capture reads on demand without an application frame queue.

The CLI owns camera, server, and OBS lifetimes and releases them on cancellation,
startup failure, or disconnection. A failed camera read stops the run with an error;
reconnect the camera and restart rather than silently changing devices. The frame
processor owns only its keyboard state. `CameraProtocol` keeps processing independent
of hardware; device discovery and capture construction are injectable for tests.

The end-to-end capture test uses OpenCV image-sequence capture, the actual YOLO model,
board stabilization, HTTP/WebSocket servers, and a Chrome-rendered overlay. Hardware
and OBS coexistence still need verification on each target machine, especially Windows.

Repo Layout
-----------

- `src/floptician/`: core application package
- `scripts/`: utility scripts for capture experiments, overlays, dataset generation, and data prep
- `models/`: tracked runtime model weights
- `resources/`: committed assets plus a large amount of local training/data material
- `output/`: generated run output, ignored by Git

Known Caveats
-------------

- This repo mixes application code with training and experimentation assets.
- Many scripts in `scripts/` are utility-only and are not yet normalized for portability.
- Some local workflows depend on files under `resources/` that are intentionally not fully tracked.
