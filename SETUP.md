# Floptician Setup Guide

This guide walks you through setting up Floptician on a fresh Windows machine. No programming experience required — just follow each step in order.

---

## Step 1: Install Python

1. Go to <https://www.python.org/downloads/>
2. Click the big **"Download Python 3.x.x"** button
3. Run the installer
4. **IMPORTANT: Check the box that says "Add python.exe to PATH"** at the bottom of the first screen — this is easy to miss and everything breaks without it
5. Click **Install Now**
6. When it finishes, open a **new** Command Prompt (Start Menu > type `cmd` > Enter) and type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`. If you see the Microsoft Store open instead, see [Troubleshooting](#microsoft-store-opens-instead-of-python).

## Step 2: Install Git

1. Go to <https://git-scm.com/download/win>
2. Download the **64-bit Git for Windows Setup**
3. Run the installer — the defaults are fine. Make sure **Git LFS** stays checked (it usually is by default)
4. When it finishes, open a **new** Command Prompt and type:
   ```
   git --version
   ```
   You should see something like `git version 2.x.x.windows.1`.

## Step 3: NVIDIA GPU drivers (for fast detection)

Floptician uses your NVIDIA GPU to detect cards in real time. Without it, detection runs on CPU and is much slower.

1. Right-click the Desktop and look for **"NVIDIA Control Panel"** or check if there's an NVIDIA icon in your system tray — if either exists, your drivers are already installed
2. If not, go to <https://www.nvidia.com/drivers> and download the latest driver for your GPU
3. To verify, open a Command Prompt and run:
   ```
   nvidia-smi
   ```
   You should see a table showing your GPU name, driver version, and CUDA version. The CUDA version should be **12.4 or higher**.

> **No NVIDIA GPU?** Floptician still works — it just falls back to CPU, which is slower. You can skip this step.

## Step 4: Clone the repo

Open a Command Prompt and run:

```
cd %USERPROFILE%\Desktop
git clone https://github.com/tpitts/floptician.git
```

This downloads the project to your Desktop. It's about **300 MB** because it includes the YOLO model files — this is normal.

When it's done you'll have a `floptician` folder on your Desktop.

## Step 5: Run setup.bat

1. Open the `floptician` folder on your Desktop
2. **Double-click `setup.bat`**
3. A black terminal window will open and start installing things — this takes a few minutes the first time
4. Wait until you see **"Setup complete!"**
5. Press any key to close the window

## Step 6: Set up OBS — Video Source + Virtual Camera

Floptician doesn't read from the physical webcam directly. Instead, OBS captures the webcam and outputs a **Virtual Camera** that Floptician reads from. This lets you position and crop the camera feed in OBS.

1. Open **OBS Studio**
2. In the **Sources** panel at the bottom, click the **+** button
3. Select **Video Capture Device**
4. Name it whatever you want (e.g. "Poker Camera"), click OK
5. Select your physical webcam from the **Device** dropdown, click OK
6. Position and crop the source as needed so the poker table fills the frame
7. In the **Controls** dock (bottom-right), click **Start Virtual Camera**

The Virtual Camera is now active — Floptician will use it as its input.

## Step 7: Set up OBS WebSocket

Floptician sends the card overlay back into OBS through a WebSocket connection. This is built into OBS — you just need to enable it.

1. In OBS, go to **Tools > WebSocket Server Settings**
2. Check **Enable WebSocket Server**
3. Note whether **Enable Authentication** is checked:
   - If authentication is **off**: you're done, skip to Step 9
   - If authentication is **on**: note the password shown (or click "Show Connect Info" to see it), then continue to Step 8
4. Click **OK**

## Step 8: Edit config.yaml (only if OBS has a WebSocket password)

If OBS WebSocket authentication is enabled:

1. In the `floptician` folder, right-click **config.yaml** and choose **Open with > Notepad**
2. Find the line that says:
   ```
   password: ''
   ```
3. Type your OBS WebSocket password between the quotes:
   ```
   password: 'your-password-here'
   ```
4. Save the file (Ctrl+S) and close Notepad

If OBS WebSocket authentication is disabled, skip this step — the default empty password works fine.

## Step 9: Run the app

1. **Double-click `run.bat`**
2. The app will list available cameras — type the number for **"OBS Virtual Camera"** and press Enter
3. Floptician starts detecting cards and automatically creates a browser source overlay in your OBS scene

To stop, close the terminal window or press Ctrl+C.

---

## Getting Updates

When there's a new version:

1. Open the `floptician` folder
2. **Double-click `update.bat`**
3. Wait for it to finish, press any key to close

That's it — your code and dependencies are up to date. Your `config.yaml` won't be overwritten.

---

## Troubleshooting

### Microsoft Store opens instead of Python

Windows sometimes redirects `python` to the Microsoft Store. To fix this:

1. Open **Settings > Apps > Advanced app settings > App execution aliases**
2. Turn **off** the toggles for "App Installer — python.exe" and "App Installer — python3.exe"
3. Open a **new** Command Prompt and try `python --version` again

### "python" is not recognized

Python wasn't added to PATH during installation. The easiest fix:

1. Uninstall Python from **Settings > Apps**
2. Re-install it and make sure you check **"Add python.exe to PATH"**

### No cameras found

- Make sure OBS is running and **Virtual Camera is started** (Step 6)
- Try restarting OBS, then run `run.bat` again

### OBS connection failed / "Could not connect to OBS"

- Make sure OBS is running
- Check that WebSocket Server is enabled (Step 7)
- If you set a password in OBS, make sure it matches `config.yaml` (Step 8)
- Make sure the port in `config.yaml` matches OBS (default is `4455`)

### Detection is very slow / not using GPU

If setup.bat said "No NVIDIA GPU detected" but you do have an NVIDIA GPU:

1. Make sure your NVIDIA drivers are up to date (Step 3)
2. Open a Command Prompt and run `nvidia-smi` — if it's not found, your drivers aren't installed
3. After fixing drivers, re-run `setup.bat` — it will reinstall PyTorch with CUDA

To check if PyTorch is using your GPU, open a Command Prompt in the floptician folder and run:
```
venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available())"
```
This should print `True`. If it prints `False`, PyTorch was installed without CUDA — re-run `setup.bat`.

### Model files missing / tiny .pt files / detection not working

The YOLO model files are stored with Git LFS. If they didn't download properly, they'll be tiny text files instead of ~100 MB model files. To fix:

1. Open a Command Prompt
2. Run:
   ```
   cd %USERPROFILE%\Desktop\floptician
   git lfs install
   git lfs pull
   ```

### Windows Firewall popup

When you first run Floptician, Windows may ask to allow network access. Click **Allow** — the app needs to communicate locally with OBS via WebSocket.

### Port already in use

If you see an error about port 8000 or 9001 being in use, another program (or a previous Floptician instance) is using that port. Close any old Floptician windows and try again.
