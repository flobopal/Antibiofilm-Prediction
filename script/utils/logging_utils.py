import logging
import sys

def setup_logging(log_file: str = None, level=logging.INFO):
    """
    Configures logging with a simple format. Can log to console and optionally to a file.

    Args:
        log_file (str): Path to the file where logs will be saved. If None, logs only to console.
        level: Logging level (logging.INFO, logging.DEBUG, etc).
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove all existing handlers
    while root.handlers:
        root.removeHandler(root.handlers[0])

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)