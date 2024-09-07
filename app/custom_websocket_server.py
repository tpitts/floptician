from websocket_server import WebsocketServer
import json
import logging
import threading

# Get the logger from the main module
logger = logging.getLogger(__name__)

class WebSocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server = WebsocketServer(host=host, port=port)
        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_client_left(self.client_left)
        self.server.set_fn_message_received(self.message_received)
        self.last_message = None
        self._is_running = threading.Event()
        self.server_thread = None

    def start(self):
        """
        Starts the WebSocket server.
        """
        logger.debug(f"WebSocket server starting on {self.host}:{self.port}")
        try:
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
        except Exception as e:
            logger.error(f"Error starting WebSocket server: {e}", exc_info=True)
            self._is_running.clear()

    def _run_server(self):
        try:
            self._is_running.set()
            self.server.run_forever()
        except Exception as e:
            logger.error(f"Error in WebSocket server: {e}", exc_info=True)
        finally:
            self._is_running.clear()

    def new_client(self, client, server):
        logger.debug(f"New client connected and was given id {client['id']}")
        if self.last_message:
            self.send_message(self.last_message, client)

    def client_left(self, client, server):
        logger.debug(f"Client({client['id']}) disconnected")

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

    def stop(self):
        """
        Stops the WebSocket server.
        """
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self._is_running.clear()
        logger.debug("WebSocket Server stopped")

    def is_running(self):
        """
        Check if the WebSocket server is running.

        Returns:
            bool: True if the server is running, False otherwise.
        """
        return self._is_running.is_set() and (self.server_thread is not None and self.server_thread.is_alive())