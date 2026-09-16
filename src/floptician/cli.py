from __future__ import annotations

import logging
import os
import platform
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import questionary
import typer
from rich.console import Console
from rich.table import Table

from floptician.config_utils import load_config, validate_config
from floptician.exceptions import (
    CameraError,
    ConfigurationError,
    ModelLoadError,
    OBSConnectionError,
    ServerStartupError,
)
from floptician.logging_config import configure_logging
from floptician.models import AppConfig, CaptureMode, FrameProcessorState

app = typer.Typer(
    name="floptician",
    help="Real-time poker card detection with YOLOv8, OBS, and WebSocket overlay.",
    add_completion=False,
)

console = Console()
logger = logging.getLogger(__name__)
MENU_STYLE = questionary.Style(
    [
        ("highlighted", "reverse bold"),
        ("pointer", "bold"),
        ("answer", "bold"),
    ]
)

BANNER = r"""[bold green]
$$$$$$$$\ $$\                      $$\     $$\           $$\
$$  _____|$$ |                     $$ |    \__|          \__|
$$ |      $$ | $$$$$$\   $$$$$$\ $$$$$$\   $$\  $$$$$$$\ $$\  $$$$$$\  $$$$$$$\
$$$$$\    $$ |$$  __$$\ $$  __$$\\_$$  _|  $$ |$$  _____|$$ | \____$$\ $$  __$$\
$$  __|   $$ |$$ /  $$ |$$ /  $$ | $$ |    $$ |$$ /      $$ | $$$$$$$ |$$ |  $$ |
$$ |      $$ |$$ |  $$ |$$ |  $$ | $$ |$$\ $$ |$$ |      $$ |$$  __$$ |$$ |  $$ |
$$ |      $$ |\$$$$$$  |$$$$$$$  | \$$$$  |$$ |\$$$$$$$\ $$ |\$$$$$$$ |$$ |  $$ |
\__|      \__| \______/ $$  ____/   \____/ \__| \_______|\__| \_______|\__|  \__|
                        $$ |
                        $$ |
                        \__|
[/bold green]"""


def _determine_torch_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif platform.system() == "Darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def _initialize_config(config_path: str | None, debug: bool) -> AppConfig:
    try:
        config = load_config(config_path)
        validate_config(config)
    except (FileNotFoundError, ConfigurationError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    if debug:
        config.debug = True
    configure_logging(debug=config.debug)

    config.platform = platform.system()

    with console.status("[bold cyan]Detecting inference device..."):
        config.torch_device = _determine_torch_device()

    config.output_dir = os.path.join(config.output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(config.output_dir, exist_ok=True)

    return config


def _select_input(inputs, input_type: str):
    names = [device if isinstance(device, str) else device.name for device in inputs]
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            index = questionary.select(
                f"Select a {input_type}",
                choices=[
                    questionary.Choice(f" {name.ljust(max(map(len, names)))} ", value=i) for i, name in enumerate(names)
                ],
                style=MENU_STYLE,
                use_shortcuts=len(inputs) <= 9,
                use_jk_keys=False,
                instruction="(↑/↓"
                + (f" or 1-{len(inputs)}" if len(inputs) <= 9 else "")
                + "; Enter selects; Ctrl+C cancels)",
            ).unsafe_ask()
        except (KeyboardInterrupt, EOFError):
            raise typer.Abort() from None
        return inputs[index]

    # Preserve numbered input for redirected stdin and non-interactive terminals.
    table = Table(title=f"Available {input_type}s")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Name", style="green")

    for i, name in enumerate(names):
        table.add_row(str(i + 1), name)

    console.print(table)

    while True:
        selection = input(f"Enter the number of the {input_type} you want to use: ")
        try:
            index = int(selection) - 1
            if 0 <= index < len(inputs):
                return inputs[index]
            else:
                console.print("[yellow]Invalid selection. Please try again.[/yellow]")
        except ValueError:
            console.print("[yellow]Invalid input. Please enter a number.[/yellow]")


@app.command()
def run(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to the config YAML file"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """Run the Floptician card detection system."""
    if console.is_terminal and console.width >= 80:
        console.print(BANNER)
    else:
        console.print("Floptician · Poker card detection", style="bold green")
    app_config = _initialize_config(config, debug)

    from floptician.board_processor import BoardProcessor
    from floptician.camera_manager import CameraManager
    from floptician.frame_processor import FrameProcessor
    from floptician.http_server import HTTPServer
    from floptician.obs_client import OBSClient
    from floptician.websocket_server import WebSocketServer

    logger.info("Platform: %s", app_config.platform)
    try:
        # The composition root owns all external resources, including failure paths.
        with ExitStack() as resources:
            if app_config.capture.mode == CaptureMode.DIRECT_WEBCAM:
                camera = resources.enter_context(CameraManager(swap_channels=app_config.capture.swap_channels))
                devices = camera.get_available_cameras()
                if not devices:
                    raise CameraError("No cameras found. Check camera permissions and connections.")
                selected = devices[0] if len(devices) == 1 else _select_input(devices, "camera")
                capture = app_config.capture
                camera.open_camera(
                    selected,
                    camera_width=capture.camera_width,
                    camera_height=capture.camera_height,
                    camera_fps=capture.camera_fps,
                )
                capture.camera_manager = camera

            logger.info("Loading model: %s", Path(app_config.yolo.model).name)
            board_processor = BoardProcessor(app_config)
            if Path(app_config.yolo.model).suffix == ".mlpackage":
                logger.info("Inference: Core ML | configured compute units: %s", app_config.yolo.coreml_compute_unit)
            else:
                logger.info("Inference: PyTorch | device: %s", app_config.torch_device)

            http_server = HTTPServer(
                app_config.host, app_config.http_port, app_config.html_file, app_config.websocket_port
            )
            http_server.start()
            resources.callback(http_server.stop)
            websocket_server = WebSocketServer(app_config.host, app_config.websocket_port)
            websocket_server.start()
            resources.callback(websocket_server.stop)
            deadline = time.monotonic() + 5
            while not (http_server.is_running() and websocket_server.is_running()):
                if time.monotonic() >= deadline:
                    raise ServerStartupError("HTTP or WebSocket server failed to start within 5 seconds")
                time.sleep(0.05)

            obs_client = OBSClient(app_config)
            resources.callback(obs_client.disconnect)
            version = obs_client.get_version()
            logger.info("Connected to OBS WebSocket %s", version.obs_web_socket_version)
            obs_client.setup_overlay()
            if app_config.capture.mode == CaptureMode.OBS_WEBSOCKET:
                webcams = obs_client.get_webcams()
                if not webcams:
                    raise CameraError("No camera sources found in OBS")
                app_config.capture.selected_webcam = (
                    webcams[0] if len(webcams) == 1 else _select_input(webcams, "webcam")
                )

            logger.info("Ready | detection target %.2f fps | q or Ctrl+C to quit", app_config.capture.fps)
            logger.info("Overlay: http://%s:%s/", app_config.host, app_config.http_port)
            processor = FrameProcessor(app_config, obs_client, board_processor, websocket_server)
            processor.run()
            if processor.state == FrameProcessorState.FAILED:
                raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except (KeyboardInterrupt, typer.Abort):
        logger.info("Capture cancelled")
        raise typer.Exit(code=130) from None
    except (CameraError, ModelLoadError, OBSConnectionError, ServerStartupError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from None
    except Exception:
        logger.exception("Unexpected capture error")
        raise typer.Exit(code=1) from None


@app.command("list-cameras")
def list_cameras() -> None:
    """List available cameras on this system."""
    from floptician.camera_manager import CameraManager

    camera_manager = CameraManager()
    try:
        cameras = camera_manager.get_available_cameras()
    except CameraError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if not cameras:
        console.print("[yellow]No cameras found.[/yellow]")
        raise typer.Exit()

    table = Table(title="Available Cameras")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="green")

    for cam in cameras:
        table.add_row(str(cam.index), cam.name)

    console.print(table)


@app.command("validate-config")
def validate_config_cmd(
    config: str | None = typer.Option(None, "--config", "-c", help="Path to the config YAML file"),
) -> None:
    """Validate a configuration file."""
    try:
        app_config = load_config(config)
    except (FileNotFoundError, ConfigurationError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    try:
        validate_config(app_config)
    except ConfigurationError as e:
        console.print(f"[red]Validation error:[/red] {e}")
        raise typer.Exit(code=1) from None

    console.print(f"[green]Config is valid![/green] Loaded from: {app_config.config_path}")
    console.print(f"  YOLO model: {app_config.yolo.model}")
    console.print(f"  Capture mode: {app_config.capture.mode.value}")
    console.print(f"  OBS: {app_config.obs.host}:{app_config.obs.port}")


if __name__ == "__main__":
    app()
