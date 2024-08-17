"""
obs_client.py: OBS WebSocket Client for Floptician

This module provides an OBSClient class for interacting with OBS Studio via the obs-websocket protocol,
using the obsws_python library.

Key points:
1. Install dependencies: pip install obsws_python
2. Ensure OBS Studio has the obs-websocket plugin installed and configured.
3. The OBSClient class methods use snake_case, matching obsws_python's conventions.
4. Most method calls use positional arguments rather than keyword arguments, following
   obsws_python's implementation. For example:
   self.client.set_input_settings("Floptician", settings, True)
   instead of:
   self.client.set_input_settings(inputName="Floptician", inputSettings=settings, overlay=True)

Main OBS WebSocket API calls used:
- get_scene_list()
- get_current_program_scene()
- get_input_list()
- set_input_settings()
- get_scene_item_list()
- set_scene_item_enabled()
- create_input()
- get_source_screenshot()

For full OBS WebSocket API documentation, visit:
https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md

This client is tailored for the Floptician project, focusing on overlay setup and webcam frame capture.
"""

import base64
import obsws_python as obs
import logging

# Import the logger from the main script
logger = logging.getLogger('echelon')

class OBSClient:
    def __init__(self, config):
        """
        Initialize the OBS WebSocket client.
        
        :param config: A dictionary containing OBS connection details and other settings.
        """
        self.client = obs.ReqClient(
            host=config['obs']['host'],
            port=config['obs']['port'],
            password=config['obs']['password']
        )
        self.config = config
        logger.debug("OBSClient initialized")

    def get_version(self):
        """Get the OBS and WebSocket versions."""
        version = self.client.get_version()
        logger.debug(f"OBS version: {version.obs_version}, WebSocket version: {version.obs_web_socket_version}")
        return version

    def get_webcams(self):
        inputs = self.client.get_input_list().inputs
        webcams = [input['inputName'] for input in inputs if input['inputKind'] == 'dshow_input']
        logger.debug(f"Detected webcams: {webcams}")
        return webcams

    def capture_frame(self, webcam_source):
        logger.debug(f"Capturing frame from {webcam_source}")
        response = self.client.get_source_screenshot(
            webcam_source,
            "png",
            self.config['capture']['width'],
            self.config['capture']['height'],
            100
        )
        
        if not hasattr(response, 'image_data'):
            logger.error("'image_data' not found in the response")
            return None

        base64_data = response.image_data.split(',', 1)[1]
        image_data = base64.b64decode(base64_data)
        
        logger.debug(f"Frame captured, size: {len(image_data)} bytes")
        return image_data

    def setup_overlay(self):
        """
        Set up or update the Floptician overlay in OBS.
        
        This method updates the URL if the "Floptician" source already exists,
        refreshes the browser source, and then ensures the source is visible.
        If it doesn't exist, it creates a new browser source.
        """
        scenes = self.client.get_scene_list().scenes
        scene_names = [scene['sceneName'] for scene in scenes]
        logger.info(f"Available scenes: {scene_names}")

        current_scene = self.client.get_current_program_scene().current_program_scene_name
        logger.info(f"Current scene: {current_scene}")

        inputs = self.client.get_input_list().inputs
        input_names = [input['inputName'] for input in inputs]

        if "Floptician" in input_names:
            logger.info("Floptician source already exists. Updating its URL, refreshing, and making it visible.")
            settings = {
                "url": f"http://{self.config['host']}:{self.config['http_port']}/"
            }
            self.client.set_input_settings("Floptician", settings, True)
            logger.info(f"Updated Floptician URL to http://{self.config['host']}:{self.config['http_port']}/")

            Floptician_input = next(input for input in inputs if input['inputName'] == 'Floptician')
            logger.debug(f"Floptician input structure: {Floptician_input}")

            # Refresh the browser source
            try:
                self.client.press_input_properties_button("Floptician", "refreshnocache")
                logger.debug("Refreshed Floptician browser source.")
            except Exception as e:
                logger.warning(f"Failed to refresh Floptician browser source: {str(e)}")

            scene_items = self.client.get_scene_item_list(current_scene).scene_items

            Floptician_item = next((item for item in scene_items if item['inputKind'] == 'browser_source' and item['sourceName'] == 'Floptician'), None)

            if Floptician_item:
                item_id = Floptician_item.get('sceneItemId')
                if item_id is not None:
                    self.client.set_scene_item_enabled(current_scene, item_id, True)
                    logger.info("Set Floptician source to be visible.")
                else:
                    logger.warning("Could not find sceneItemId for Floptician")
            else:
                logger.warning("Could not find Floptician in the current scene items")

        else:
            logger.info("Creating new Floptician overlay")
            settings = {
                "url": f"http://{self.config['host']}:{self.config['http_port']}/",
                "width": 720,
                "height": 1280
            }
            self.client.create_input(current_scene, "Floptician", "browser_source", settings, True)
            logger.info("Floptician overlay added to the current scene.")

    def disconnect(self):
        """Disconnect from the OBS WebSocket server."""
        self.client.disconnect()
        logger.info("Disconnected from OBS WebSocket server")