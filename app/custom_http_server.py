from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging

# Get the logger from the main module
logger = logging.getLogger(__name__)

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Serve overlay.html only when root URL is accessed
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
        logger.info(f"{self.client_address[0]} - - [{self.log_date_time_string()}] {format % args}")

def start_http_server(host, port, html_file):
    """
    Starts the HTTP server.

    Args:
        host (str): The host address to bind to.
        port (int): The port number to listen on.
        html_file (str): The path to the HTML file to serve.
    """
    try:
        server_address = (host, port)
        httpd = ThreadingHTTPServer(server_address, MyHandler)
        httpd.html_file_path = html_file  # Pass the HTML file path to the server instance
        logger.info(f"HTTP Server started at http://{host}:{port}")
        logger.info(f"Serving file: {html_file}")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Error starting HTTP server: {e}", exc_info=True)