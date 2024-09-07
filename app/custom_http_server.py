from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import threading

# Get the logger from the main module
logger = logging.getLogger(__name__)

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            try:
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('Expires', '0')
                self.end_headers()
                with open(self.server.html_file_path, 'rb') as file:
                    self.wfile.write(file.read())
                logger.info(f"Served {self.server.html_file_path}")
            except Exception as e:
                logger.error(f"Error serving {self.server.html_file_path}: {e}", exc_info=True)
                self.send_error(500, "Internal server error")
        else:
            self.send_error(403, "Forbidden")
            logger.warning(f"Attempted to access forbidden path: {self.path}")

    def log_message(self, format, *args):
        logger.debug(f"{self.client_address[0]} - - [{self.log_date_time_string()}] {format % args}")

class HTTPServer:
    def __init__(self, host, port, html_file):
        self.host = host
        self.port = port
        self.html_file = html_file
        self.server = None
        self._is_running = threading.Event()
        self.server_thread = None

    def start(self):
        """
        Starts the HTTP server.
        """
        try:
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
        except Exception as e:
            logger.error(f"Error starting HTTP server: {e}", exc_info=True)
            self._is_running.clear()

    def _run_server(self):
        try:
            self.server = ThreadingHTTPServer((self.host, self.port), MyHandler)
            self.server.html_file_path = self.html_file
            logger.debug(f"HTTP Server running at http://{self.host}:{self.port}")
            logger.debug(f"Serving file: {self.html_file}")
            self._is_running.set()
            self.server.serve_forever()
        except Exception as e:
            logger.error(f"Error in HTTP server: {e}", exc_info=True)
        finally:
            self._is_running.clear()
            if self.server:
                self.server.server_close()

    def stop(self):
        """
        Stops the HTTP server.
        """
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        self._is_running.clear()
        logger.debug("HTTP Server stopped")

    def is_running(self):
        """
        Check if the HTTP server is running.

        Returns:
            bool: True if the server is running, False otherwise.
        """
        return self._is_running.is_set() and (self.server_thread is not None and self.server_thread.is_alive())
