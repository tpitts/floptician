#!/bin/bash
# Fresh, rerunnable Apple Silicon install. Never touches an older checkout.
set -euo pipefail
trap 'echo "Installation stopped. Review the message above; it is safe to retry." >&2' ERR
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "Floptician requires an Apple Silicon Mac. Run Terminal without Rosetta." >&2
    exit 1
fi
mac_major="$(sw_vers -productVersion | cut -d. -f1)"
if (( mac_major < 14 )); then
    echo "macOS 14 or newer is required. Update macOS in System Settings, then retry." >&2
    exit 1
fi
if ! xcode-select -p >/dev/null 2>&1; then
    echo "Apple's command-line tools are needed for Git. Run: xcode-select --install"
    echo "Finish that installation, then run this installer again. No Homebrew is needed."
    exit 1
fi
install_dir="$HOME/Applications/Floptician"
if [[ -e "$install_dir" ]]; then
    if [[ ! -f "$install_dir/.floptician-mac-install" || ! -d "$install_dir/.git" ]]; then
        echo "$install_dir already exists and is not managed by this installer. Nothing was overwritten." >&2
        exit 1
    fi
    echo "Existing managed install found. Resuming setup (use Update Floptician for code updates)."
else
    mkdir -p "$HOME/Applications"
    staging_dir="$(mktemp -d "$HOME/Applications/.Floptician-install-XXXXXXXX")"
    GIT_LFS_SKIP_SMUDGE=1 git clone --branch master https://github.com/tpitts/floptician.git "$staging_dir"
    touch "$staging_dir/.floptician-mac-install"
    if [[ -e "$install_dir" ]]; then
        echo "Another installation appeared at $install_dir. Download left at $staging_dir." >&2; exit 1
    fi
    mv "$staging_dir" "$install_dir"
fi
/bin/bash "$install_dir/scripts/mac/manage.sh" setup
