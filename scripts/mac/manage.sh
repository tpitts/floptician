#!/bin/bash
# All operations are rooted here, independent of cwd, pyenv, or activated venvs.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
action="${1:-setup}"
case "$action" in setup|run|update|password) ;; *) echo "Use setup, run, update, or password" >&2; exit 1 ;; esac
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Use an Apple Silicon Mac and a native Terminal (not Rosetta)." >&2; exit 1
fi
if (( $(sw_vers -productVersion | cut -d. -f1) < 14 )); then
    echo "macOS 14 or newer is required. Update macOS before continuing." >&2; exit 1
fi
export PATH="$repo_dir/.tools/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export UV_PROJECT_ENVIRONMENT="$repo_dir/.venv"
export UV_PYTHON_PREFERENCE=only-managed
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH UV_PROJECT UV_WORKING_DIRECTORY FLOPTICIAN_CONFIG
lock_dir="$repo_dir/.floptician-operation.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "Floptician is already running, installing, or updating. Close it first." >&2
    echo "After a crash: if no Floptician process is running, remove $lock_dir and retry." >&2
    exit 1
fi
cleanup() { rmdir "$lock_dir" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
trap 'echo "Operation failed. Read the error above. No settings were replaced; retry Setup or Update after fixing it." >&2' ERR

environment_signature() { LC_ALL=C shasum -a 256 uv.lock pyproject.toml .python-version; }

if [[ "$action" == password ]]; then
    if [[ ! -x .venv/bin/python || ! -f config.yaml ]]; then
        echo "Complete setup first." >&2; exit 1
    fi
    uv run --no-sync python scripts/mac/configure.py --obs-password
    exit 0
fi

if [[ "$action" == run ]]; then
    if [[ ! -f .venv/floptician-mac-ready.txt ]] || \
        [[ "$(environment_signature)" != "$(cat .venv/floptician-mac-ready.txt)" ]]; then
        echo "Setup is missing or dependencies changed. Run Setup Floptician.command first." >&2; exit 1
    fi
    uv run --no-sync floptician run --config "$repo_dir/config.yaml"
    exit 0
fi

if ! xcode-select -p >/dev/null 2>&1; then
    echo "Run xcode-select --install, finish Apple's installer, then retry." >&2; exit 1
fi
if [[ -d .venv && ! -f .venv/floptician-mac-managed && ! -f .floptician-mac-environment ]]; then
    echo "Existing unmanaged .venv: refusing to overwrite it. Use a fresh installation folder." >&2; exit 1
fi
if (( $(df -Pk . | awk 'END {print $4}') < 5 * 1024 * 1024 )); then
    echo "Please free at least 5 GB of disk space before installing/updating." >&2; exit 1
fi
mkdir -p .tools/bin
if [[ ! -x .tools/bin/uv ]]; then
    echo "Installing private uv (no shell-profile changes)..."
    curl -fL --retry 3 https://astral.sh/uv/0.11.7/install.sh -o .tools/install-uv.sh
    UV_UNMANAGED_INSTALL="$repo_dir/.tools/bin" /bin/sh .tools/install-uv.sh
fi
if ! command -v git-lfs >/dev/null 2>&1; then
    echo "Installing private Git LFS..."
    curl -fL --retry 3 https://github.com/git-lfs/git-lfs/releases/download/v3.8.0/git-lfs-darwin-arm64-v3.8.0.zip -o .tools/git-lfs.zip
    echo "caff76a7d070d8160c89bc39b6e85d98f24135b6fed038a3b4de2590d25102d8  .tools/git-lfs.zip" | LC_ALL=C shasum -a 256 -c -
    unzip -oq .tools/git-lfs.zip -d .tools/lfs
    cp .tools/lfs/git-lfs-3.8.0/git-lfs .tools/bin/git-lfs
    chmod +x .tools/bin/git-lfs
fi
git lfs install --local

if [[ "$action" == update ]]; then
    if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
        echo "Tracked files have local edits. Update stopped; ask Tom to help. Nothing was reset." >&2; exit 1
    fi
    if [[ "$(git branch --show-current)" != master ]]; then
        echo "Update expects the master branch. Ask Tom to help; no branch was changed." >&2; exit 1
    fi
    origin="$(git remote get-url origin)"
    case "$origin" in https://github.com/tpitts/floptician|https://github.com/tpitts/floptician.git) ;;
        *) echo "Unexpected origin: update stopped." >&2; exit 1 ;; esac
    mkdir -p output/setup-backups
    backup_dir="$(mktemp -d "$repo_dir/output/setup-backups/update-XXXXXXXX")"
    chmod 700 "$backup_dir"
    if [[ -f config.yaml ]]; then cp -p config.yaml "$backup_dir/config.yaml"; chmod 600 "$backup_dir/config.yaml"; fi
    git rev-parse HEAD > "$backup_dir/previous-commit.txt"
    echo "Backup: $backup_dir"
    GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only origin master
    # Reload the updated setup code, rather than finishing with an old updater.
    cleanup
    trap - EXIT
    exec /bin/bash "$repo_dir/scripts/mac/manage.sh" setup
fi

echo "Installing locked Python and Core ML dependencies (first download may take several minutes)..."
touch .floptician-mac-environment
if [[ -f .venv/floptician-mac-ready.txt ]]; then
    mv -f .venv/floptician-mac-ready.txt .venv/floptician-mac-previous-ready.txt
fi
uv sync --locked --no-dev --extra coreml
# uv may recreate an empty environment during initial setup.
touch .venv/floptician-mac-managed
uv run --no-sync python scripts/mac/configure.py
echo "Downloading the selected model..."
model_include="$(uv run --no-sync python scripts/mac/configure.py --model-include)"
if [[ -n "$model_include" ]]; then git lfs pull --include "$model_include" --exclude ''; fi
uv run --no-sync floptician validate-config --config "$repo_dir/config.yaml"
uv run --no-sync python scripts/mac/configure.py --smoke-test
environment_signature > .venv/floptician-mac-ready.txt
echo "Setup complete. Open OBS, enable Tools > WebSocket Server Settings, then double-click Run Floptician.command."
echo "Choose your physical camera on Mac. Approve camera access when macOS asks."
echo "Run/Update shortcuts are in: $repo_dir"
