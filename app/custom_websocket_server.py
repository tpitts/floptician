from websocket_server import WebsocketServer
import json
import logging

# Get the logger from the main module
logger = logging.getLogger(__name__)

# WebSocket server
class WebSocketServer:
    def __init__(self, host, port):
        self.server = WebsocketServer(host=host, port=port)
        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_client_left(self.client_left)
        self.server.set_fn_message_received(self.message_received)
        self.last_message = None

    def start(self):
        logger.info(f"WebSocket server starting on {self.server.host}:{self.server.port}")
        try:
            self.server.run_forever()
        except Exception as e:
            logger.error(f"Error in WebSocket server: {e}", exc_info=True)

    def new_client(self, client, server):
        logger.info(f"New client connected and was given id {client['id']}")
        if self.last_message:
            self.send_message(self.last_message, client)

    def client_left(self, client, server):
        logger.info(f"Client({client['id']}) disconnected")

    def message_received(self, client, server, message):
        logger.debug(f"Client({client['id']}) said: {message}")

    def send_message(self, message, client=None):
        try:
            json_message = json.dumps(message)
            if client:
                self.server.send_message(client, json_message)
                logger.debug(f"Sent message to client {client['id']}: {json_message[:100]}...")
            else:
                self.server.send_message_to_all(json_message)
                logger.debug(f"Sent message to all clients: {json_message[:100]}...")
            self.last_message = message
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}", exc_info=True)