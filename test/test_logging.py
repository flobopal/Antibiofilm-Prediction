import logging
import tempfile
import os
from script.utils.logging_utils import setup_logging

def test_setup_logging_sets_correct_level():
    setup_logging(level = logging.DEBUG)
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG

def test_setup_logging_adds_handler_if_none():
    logger = logging.getLogger()
    logger.handlers = []
    setup_logging()
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)

def test_setup_logging_does_not_duplicate_handlers():
    logger = logging.getLogger()
    logger.handlers = []  # limpiar
    setup_logging()
    handler_count = len(logger.handlers)
    setup_logging()  # llamar de nuevo
    assert len(logger.handlers) == handler_count  # no se añaden más handlers

def test_setup_logging_sets_correct_formatter():
    logger = logging.getLogger()
    logger.handlers = []
    setup_logging()
    formatter = logger.handlers[0].formatter
    assert isinstance(formatter, logging.Formatter)
    assert formatter._fmt == '%(asctime)s | %(levelname)s | %(message)s'

def test_logging_writes_to_file():
    # Create temporal file
    with tempfile.NamedTemporaryFile(delete=False, mode='r+', encoding='utf-8') as tmp_log_file:
        tmp_log_path = tmp_log_file.name

    try:
        # Clean root logger
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Configure to print into that file
        setup_logging(log_file=tmp_log_path)

        # Send a message
        logging.info("Test log message")

        # Check the message is written
        with open(tmp_log_path, 'r', encoding='utf-8') as f:
            contents = f.read()

        assert "Test log message" in contents
        assert "INFO" in contents

    finally:
        os.remove(tmp_log_path)