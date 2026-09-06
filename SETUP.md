# Floptician setup

Floptician uses uv to install the Python version in `.python-version` and the
packages recorded in `uv.lock`. Run commands from the repository root.

## Windows

1. Install **Git for Windows**, including Git LFS.
2. Install **uv** using the Windows instructions at
   https://docs.astral.sh/uv/getting-started/installation/ and open a new terminal.
   Version 0.11.7 or newer is required. You do not need to install Python separately.
3. Clone the repository, then download the model weights:

   ```text
   git clone https://github.com/tpitts/floptician.git
   cd floptician
   git lfs pull
   ```

4. Double-click `setup.bat`. On the first installation it selects CUDA if
   `nvidia-smi` is available, otherwise CPU. It creates `.venv` and copies
   `config.example.yaml` only when `config.yaml` does not already exist.
   Wait for **Setup complete**. CUDA downloads several gigabytes.
5. Configure OBS as described below, then double-click `run.bat`.

You can choose the backend explicitly from Command Prompt:

```text
setup.bat cpu
setup.bat cuda
```

CUDA uses the locked PyTorch build for CUDA 12.8. Install a compatible NVIDIA
GPU driver. Setup checks CUDA availability and stops if it is unavailable;
it does not silently report success on CPU. The selected backend is saved in
`.venv/floptician-backend.txt` and reused on subsequent setup and updates.

## Apple Silicon Mac

**Migration prepared; hardware validation is still pending.** Keep the working
Mac environment until the new one has passed the tests and a live OBS session.
The locked environment targets Apple Silicon; Intel Macs are not covered.

Install Git LFS and uv using their official installation instructions, clone the
repository, then run:

```sh
git lfs pull
uv sync --locked --no-dev
cp -n config.example.yaml config.yaml
uv run --no-sync floptician validate-config
uv run --no-sync floptician run
```

The Mac PyTorch package supports CPU and MPS GPU inference. Core ML conversion is
separate work; this migration does not convert the model. If you need the existing
optional Core ML tools, add `--extra coreml` to the sync command. Do not use the
Windows CUDA extra on a Mac.

## Migrating an existing environment

Leave `venv` and `flopenv` in place as backups. They are no longer used by the
updated Windows launchers. If `.venv` already exists and was not created by the
new setup, close Floptician and rename it to an unused backup name before setup.
The Windows script refuses to overwrite an unmanaged `.venv`.

Virtual environments can contain absolute paths: to restore an old `.venv`, close
the app and put it back at its original path. Keep the corresponding old code
revision too. Do not delete the backups until the replacement works in a live
session. On Mac, apply the same backup step before `uv sync`.

Python is pinned to **3.10.20**, staying on the existing 3.10 minor version.
The runtime baseline preserves the existing Windows `flopenv` versions of
Ultralytics (**8.4.21**), NumPy, OpenCV, Pillow, and the application dependencies.
PyTorch is **2.10.0**, with torchvision **0.25.0**: these replace unavailable
nightly builds and must be treated as a tested baseline change, not an exact
reproduction of the old environment. The model weights are unchanged.

## OBS setup

1. Open OBS Studio and add your webcam as a **Video Capture Device** source.
2. Position and crop it, then select **Start Virtual Camera**. This is the
   recommended capture path; the application can also read a physical webcam.
3. Open **Tools > WebSocket Server Settings** and enable the server.
4. If authentication is enabled, put its password in `config.yaml` under
   `obs.password`, or set the `OBS_PASSWORD` environment variable.
5. Start Floptician and select **OBS Virtual Camera** when prompted. It creates
   or refreshes the Floptician browser source in the current OBS scene.

Stop with Ctrl+C. The OBS WebSocket port defaults to 4455; Floptician's separate
HTTP and overlay WebSocket ports default to 8000 and 9001. The overlay now follows
`websocket_port` in the configuration. HTTP and overlay WebSocket ports must differ.

## Updating

On Windows, double-click `update.bat`. It runs `git pull --ff-only` and then reuses
setup to install the committed lockfile and retain your backend selection.
If Git cannot fast-forward, it stops for manual resolution. Local configuration
is preserved. Setup/update installs runtime packages; developers should re-sync
with their development extras afterwards.

On Mac:

```sh
git pull --ff-only
uv sync --locked --no-dev
uv run --no-sync floptician validate-config
```

Routine launch uses `--no-sync`, so it does not install packages or upgrade
versions. After pulling code manually, sync before launching. Windows launch
also checks that the lockfile matches the project metadata.

## Development and validation

Use the same backend choice every time you sync a Windows development environment:

```text
uv sync --locked --extra windows --extra cuda --extra browser-test
```

Substitute `--extra cpu` for `--extra cuda` for CPU testing. On Mac:

```sh
uv sync --locked --extra browser-test
```

The `dev` dependency group is installed by default. Full validation requires
**Google Chrome** plus the tracked model and screenshot fixtures:

```text
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync python -m pytest -q --require-all-tests
```

`--require-all-tests` fails if any test is skipped. A successful full run must
include the real-model screenshot tests and browser tests. These checks use
isolated browsers and mocked OBS boundaries; they do not replace a live OBS
session. Ordinary `python -m pytest` can skip missing optional prerequisites and
is not the full acceptance check. Use `-m "not screenshot" --ignore=tests/test_overlay.py`
for a deliberately smaller test run.

Dependency upgrades are deliberate: edit the relevant requirement in
`pyproject.toml`, run `uv lock`, inspect the lockfile diff, and re-run validation
before rollout. Do not hand-edit `uv.lock` or use `uv sync --all-extras`: CPU and
CUDA extras are mutually exclusive. Supported lockfile targets are Windows x64
and Apple Silicon macOS, with Python 3.10-3.12; the standard setup uses 3.10.20.

## Troubleshooting

- **uv not found:** open a new terminal after installation and check `uv --version`.
- **Existing .venv needs migration:** follow the backup instructions above. A
  failed initial install can also leave a partial `.venv`; preserve/rename it and retry.
- **CUDA unavailable:** update the driver, or explicitly use `setup.bat cpu`.
- **Missing or tiny model file:** run `git lfs install` and `git lfs pull`.
- **No cameras:** start OBS Virtual Camera before launching Floptician.
- **OBS connection failure:** check OBS is open and the configured host, port,
  and password match its WebSocket settings.
- **Port already in use:** stop the old instance or choose unused ports in config.
- **Invalid configuration:** run `floptician validate-config` through
  `uv run --no-sync` and correct the setting named in the error.
