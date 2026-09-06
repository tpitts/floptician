from __future__ import annotations

import logging
import os
import platform
import time
from datetime import datetime

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from floptician.config_utils import load_config, validate_config
from floptician.exceptions import (
    ConfigurationError,
    ModelLoadError,
    OBSConnectionError,
    ServerStartupError,
)
from floptician.logging_config import configure_logging
from floptician.models import AppConfig, CaptureMode

app = typer.Typer(
    name="floptician",
    help="Real-time poker card detection with YOLOv8, OBS, and WebSocket overlay.",
    add_completion=False,
)

console = Console()
logger = logging.getLogger(__name__)

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
    table = Table(title=f"Available {input_type}s")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Name", style="green")

    for i, input_device in enumerate(inputs):
        name = (
            input_device if isinstance(input_device, str) else input_device.get("camera_name", f"{input_type} {i + 1}")
        )
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
    rprint(BANNER)

    app_config = _initialize_config(config, debug)

    if app_config.debug:
        logger.debug("Debug mode is enabled.")
    logger.info(f"Platform: {app_config.platform}")
    logger.info(f"Inference Device: {app_config.torch_device}")

    # Lazy imports for heavy deps
    from floptician.board_processor import BoardProcessor
    from floptician.camera_manager import CameraManager
    from floptician.frame_processor import FrameProcessor
    from floptician.http_server import HTTPServer
    from floptician.obs_client import OBSClient
    from floptician.websocket_server import WebSocketServer

    # Start servers
    http_server = None
    websocket_server = None
    try:
        http_server = HTTPServer(app_config.host, app_config.http_port, app_config.html_file, app_config.websocket_port)
        http_server.start()

        websocket_server = WebSocketServer(app_config.host, app_config.websocket_port)
        websocket_server.start()

        start_time = time.time()
        http_running = ws_running = False

        while time.time() - start_time < 5:
            if not http_running and http_server.is_running():
                logger.info("HTTP server is running")
                http_running = True
            if not ws_running and websocket_server.is_running():
                logger.info("WebSocket server is running")
                ws_running = True
            if http_running and ws_running:
                break
            time.sleep(0.1)

        if not http_running:
            raise RuntimeError("HTTP server failed to start within the timeout period")
        if not ws_running:
            raise RuntimeError("WebSocket server failed to start within the timeout period")

    except ServerStartupError as e:
        logger.error(f"Server startup error: {e}")
        if http_server:
            http_server.stop()
        if websocket_server:
            websocket_server.stop()
        raise typer.Exit(code=1) from None
    except Exception as e:
        logger.error(f"Error starting servers: {e}", exc_info=True)
        if http_server:
            http_server.stop()
        if websocket_server:
            websocket_server.stop()
        raise typer.Exit(code=1) from None

    # Initialize board processor (loads YOLO model)
    try:
        with console.status("[bold cyan]Loading YOLO model..."):
            board_processor = BoardProcessor(app_config)
        logger.info(
            f"BoardProcessor started | YOLO model: {app_config.yolo.model} | Inference: {app_config.torch_device}"
        )
    except ModelLoadError as e:
        logger.error(f"Error loading YOLO model: {e!s}")
        http_server.stop()
        websocket_server.stop()
        raise typer.Exit(code=1) from None

    # Connect to OBS
    try:
        obs_client = OBSClient(app_config)
        version_info = obs_client.get_version()
        logger.info(
            f"Connected to OBS WebSocket API {app_config.obs.host}:{app_config.obs.port} "
            f"using version: {version_info.obs_web_socket_version}"
        )
        obs_client.setup_overlay()
    except OBSConnectionError as e:
        logger.error(str(e))
        http_server.stop()
        websocket_server.stop()
        raise typer.Exit(code=1) from None

    # Set up capture method
    if app_config.capture.mode == CaptureMode.OBS_WEBSOCKET:
        webcams = obs_client.get_webcams()
        if not webcams:
            logger.error("No webcams found in OBS. Exiting...")
            http_server.stop()
            websocket_server.stop()
            raise typer.Exit(code=1)
        app_config.capture.selected_webcam = webcams[0] if len(webcams) == 1 else _select_input(webcams, "webcam")

    elif app_config.capture.mode == CaptureMode.DIRECT_WEBCAM:
        camera_manager = CameraManager(swap_channels=app_config.capture.swap_channels)
        cameras = camera_manager.get_available_cameras()
        if not cameras:
            logger.error("No webcams found. Exiting...")
            http_server.stop()
            websocket_server.stop()
            raise typer.Exit(code=1)

        selected_camera = cameras[0] if len(cameras) == 1 else _select_input(cameras, "camera")
        if not camera_manager.open_camera(selected_camera["camera_index"]):
            logger.error(f"Failed to open camera {selected_camera['camera_name']}. Exiting...")
            http_server.stop()
            websocket_server.stop()
            raise typer.Exit(code=1)

        camera_manager.set_resolution(app_config.capture.width, app_config.capture.height)
        app_config.capture.camera_manager = camera_manager

    # Run frame processor
    logger.info("Starting frame processor...")
    frame_processor = FrameProcessor(app_config, obs_client, board_processor, websocket_server)

    try:
        frame_processor.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error: {e!s}", exc_info=True)
    finally:
        logger.info("Exiting capture loop")
        http_server.stop()
        websocket_server.stop()


@app.command("list-cameras")
def list_cameras() -> None:
    """List available cameras on this system."""
    from floptician.camera_manager import CameraManager

    camera_manager = CameraManager()
    cameras = camera_manager.get_available_cameras()

    if not cameras:
        console.print("[yellow]No cameras found.[/yellow]")
        raise typer.Exit()

    table = Table(title="Available Cameras")
    table.add_column("Index", style="cyan", justify="right")
    table.add_column("Name", style="green")

    for cam in cameras:
        table.add_row(str(cam["camera_index"]), cam["camera_name"])

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
