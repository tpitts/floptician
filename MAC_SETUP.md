# Floptician on an Apple Silicon Mac

Requires an **M1/M2/M3 or newer Mac and macOS 14 or newer**. Check Apple menu →
About This Mac. Keep the old Floptician installation until the new one works.
Allow at least **5 GB free space** and a reliable internet connection.

## First installation

Open Terminal, paste this whole block, then press Enter:

```sh
floptician_installer="$(mktemp -t floptician-install)" && curl -fL --retry 3 https://raw.githubusercontent.com/tpitts/floptician/master/install-mac.sh -o "$floptician_installer" && /bin/bash "$floptician_installer"
```

This downloads and runs this repository's installer. It installs into
`~/Applications/Floptician`, **not your old Floptician folder**. It does not change
your shell profile, Homebrew, pyenv, or old virtual environment.

If it asks for Apple's command-line tools, run `xcode-select --install`, finish
Apple's installation window, then paste the original command again.

During setup:

1. Wait for the Python/dependency and model downloads; the first run takes a while.
2. If you have the old `config.yaml`, drag it into Terminal when asked and press
   Enter. Otherwise press Enter to start fresh. The old file is not changed.
3. Open OBS → **Tools → WebSocket Server Settings**. Enable the server and find
   its password under **Show Connect Info**. Enter it when asked (typing is hidden).
4. Wait for **Setup complete**. An OBS-not-ready message is not a failed Python/model
   installation; fix the OBS connection before your first run.

Setup imports OBS credentials, detection rate/thresholds and server ports. It
deliberately replaces obsolete model/overlay paths and capture selection with the
current model and native camera capture. It does not import old camera indexes,
channel swaps, or screenshot dimensions. Imported originals are backed up under
`output/setup-backups/`, and the new settings file is readable only by your user.

## Every time you run

1. Open OBS with your camera source and usual scene.
2. In Finder press **Shift-Command-G**, enter `~/Applications/Floptician`, then
   double-click **Run Floptician.command**. You can make a Finder alias for the Desktop.
3. Select your **physical camera** with arrows/Enter or its menu number. You do not
   need OBS Virtual Camera for this Mac setup.
4. Approve macOS camera access when asked. If blocked, check System Settings →
   Privacy & Security → Camera for Terminal, then quit and reopen the launcher.
5. Press **q** or **Ctrl+C** to stop before closing the Terminal window.

The application creates/refreshes the Floptician browser source in the current OBS
scene. Check both camera and overlay in OBS before using it for a session.

## When Tom tells you an update is ready

Close Floptician, then double-click **Update Floptician.command**. Wait for
**Setup complete** before reopening Run. OBS can remain open.

Updates back up `config.yaml` and record the previous Git revision, pull only
fast-forward changes from `master`, install the exact locked dependencies, and
download the selected model. Your settings are not overwritten. Local code edits
or diverged Git history stop the update for Tom to resolve; nothing is reset.

If a download fails, retry Update or Setup after fixing the connection. This is
not an automatic rollback system: do not try to run a partially updated install.
Run refuses an unprepared dependency set. Keep the 2024 folder as your initial fallback.

## Fixing setup

- **Wrong OBS password:** open **Setup Floptician.command**, choose **2**, enter
  the password, then launch again. A backup is made before changing the config.
- **Repair/check dependencies and model:** choose **1** in Setup. Existing
  `config.yaml` stays untouched; the model and OBS connection are checked again.
- **An update reports local edits:** send Tom the error; do not discard files.
- **Already running:** quit the other Floptician window. After a crash, ask Tom to
  check for a stale `.floptician-operation.lock` directory. Do not delete it while
  a process is active.
- **Existing unmanaged environment/folder:** do not overwrite it. Ask Tom to help
  or use the fresh-install command above.
- **macOS blocks a launcher:** follow the normal Privacy & Security approval for
  the file you intentionally downloaded. Do not disable Gatekeeper globally.

Do not send your `config.yaml` publicly: it contains the OBS password.

## For Tom / development

`model: auto` follows the recommended Core ML model on Apple Silicon and PyTorch
model on Windows. An explicit model path stays pinned. Missing settings use code
defaults; updates do not merge or replace existing values. Breaking settings
changes need a specific, backed-up migration when they arise.

Launchers use an isolated uv-managed environment and runtime-only dependencies.
They intentionally refuse to adopt your existing development `.venv`. Continue
using normal uv commands in the development checkout; test the friend workflow
in a separate clone. New model releases require updating `DEFAULT_YOLO_MODEL`,
tracking the model in Git LFS, and running the full tests and Core ML validation
before pushing. Model conversion is never part of the friend's installation.

Installer tools: private uv 0.12.10; Git LFS 3.8.0 downloaded with a pinned SHA-256
when Git LFS is absent. Git LFS configuration is repository-local. The code and
installer come from this GitHub repository over HTTPS; this is not a signed app
distribution system. Updating `master` publishes changes to these two users.
