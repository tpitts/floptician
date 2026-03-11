Floptician
==========

Floptician is a poker table card-detection project built around live capture, YOLO-based recognition, and OBS/browser overlay output.

Run
---

The current entrypoint is:

```bash
python main.py
```

Runtime Notes
-------------

- Install dependencies with `pip install -r requirements.txt`.
- Copy `config.example.yaml` to `config.yaml` for machine-local settings.
- Local runtime configuration is read from `config.yaml` by default.
- You can also point the app at another config file with `python main.py --config path/to/config.yaml` or `FLOPTICIAN_CONFIG=...`.
- `config.yaml` is intentionally ignored and is expected to stay local to each machine.
- Model paths in config point at files under `models/`.
- `output/` is used for generated run artifacts.
- If you use OBS capture mode, OBS with its WebSocket server must be available locally.

Dependencies
------------

- `pip install -r requirements.txt` is the default install path.
- On Windows, `pygrabber` is installed automatically for better camera enumeration.
- On macOS, the current camera path uses FFmpeg via `imageio-ffmpeg`; PyObjC is not required.
- If you want to use `.mlpackage` / CoreML models, install `coremltools` separately.

Repo Layout
-----------

- `app/`: runtime modules used by `main.py`
- `scripts/`: utility scripts for capture experiments, overlays, dataset generation, and data prep
- `models/`: tracked runtime model weights
- `resources/`: committed assets plus a large amount of local training/data material
- `output/`: generated run output, ignored by Git

Known Caveats
-------------

- This repo mixes application code with training and experimentation assets.
- Many scripts in `scripts/` are utility-only and are not yet normalized for portability.
- Some local workflows depend on files under `resources/` that are intentionally not fully tracked.
