from __future__ import annotations

import base64
import logging

import obsws_python as obs

from floptician.exceptions import OBSConnectionError
from floptician.models import AppConfig

logger = logging.getLogger(__name__)


class OBSClient:
    def __init__(self, config: AppConfig):
        try:
            self.client = obs.ReqClient(
                host=config.obs.host,
                port=config.obs.port,
                password=config.obs.password,
            )
        except ConnectionRefusedError as e:
            raise OBSConnectionError(
                f"Unable to connect to OBS at {config.obs.host}:{config.obs.port}. "
                "Ensure OBS is running and the WebSocket server is enabled."
            ) from e
        except Exception as e:
            raise OBSConnectionError(f"Unexpected error connecting to OBS: {e}") from e
        self.config = config
        logger.debug("OBSClient initialized")

    def get_version(self):
        version = self.client.get_version()
        logger.debug(f"OBS version: {version.obs_version}, WebSocket version: {version.obs_web_socket_version}")
        return version

    def get_webcams(self):
        inputs = self.client.get_input_list().inputs
        webcams = [inp["inputName"] for inp in inputs if inp["inputKind"] == "dshow_input"]
        logger.debug(f"Detected webcams: {webcams}")
        return webcams

    def capture_frame(self, webcam_source):
        logger.debug(f"Capturing frame from {webcam_source}")
        response = self.client.get_source_screenshot(
            webcam_source,
            "png",
            self.config.capture.width,
            self.config.capture.height,
            100,
        )

        if not hasattr(response, "image_data"):
            logger.error("'image_data' not found in the response")
            return None

        base64_data = response.image_data.split(",", 1)[1]
        image_data = base64.b64decode(base64_data)

        logger.debug(f"Frame captured, size: {len(image_data)} bytes")
        return image_data

    def setup_overlay(self):
        scenes = self.client.get_scene_list().scenes
        scene_names = [scene["sceneName"] for scene in scenes]
        logger.debug(f"Available scenes: {scene_names}")

        current_scene = self.client.get_current_program_scene().current_program_scene_name
        logger.debug(f"Current scene: {current_scene}")

        inputs = self.client.get_input_list().inputs
        input_names = [inp["inputName"] for inp in inputs]

        overlay_url = f"http://{self.config.host}:{self.config.http_port}/"

        if "Floptician" in input_names:
            logger.debug("Floptician source already exists. Updating its URL, refreshing, and making it visible.")
            settings = {"url": overlay_url}
            self.client.set_input_settings("Floptician", settings, True)
            logger.debug(f"Updated Floptician URL to {overlay_url}")

            try:
                self.client.press_input_properties_button("Floptician", "refreshnocache")
                logger.debug("Refreshed Floptician browser source.")
            except Exception as e:
                logger.warning(f"Failed to refresh Floptician browser source: {e!s}")

            scene_items = self.client.get_scene_item_list(current_scene).scene_items

            floptician_item = next(
                (
                    item
                    for item in scene_items
                    if item["inputKind"] == "browser_source" and item["sourceName"] == "Floptician"
                ),
                None,
            )

            if floptician_item:
                item_id = floptician_item.get("sceneItemId")
                if item_id is not None:
                    self.client.set_scene_item_enabled(current_scene, item_id, True)
                    logger.debug("Set Floptician source to be visible.")
                else:
                    logger.warning("Could not find sceneItemId for Floptician")
            else:
                logger.warning("Could not find Floptician in the current scene items")

        else:
            logger.debug("Creating new Floptician overlay")
            settings = {
                "url": overlay_url,
                "width": 720,
                "height": 1280,
            }
            self.client.create_input(current_scene, "Floptician", "browser_source", settings, True)

        logger.info(f"Floptician OBS overlay active on scene: {current_scene}")

    def disconnect(self):
        self.client.disconnect()
        logger.info("Disconnected from OBS WebSocket server")
