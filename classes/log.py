import os
import yaml
import logging

FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class InfoFilter(logging.Filter):
    def filter(self, record):
        """Filter to allow only logs below WARNING level (INFO and DEBUG)."""
        return record.levelno < logging.WARNING

def safe_get(obj, path):
    for key in path:
        try:
            if isinstance(obj, dict):
                obj = obj[key]
            elif isinstance(obj, list):
                obj = obj[int(key)]
            else:
                return None
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return obj


def setup_logging() -> None:
    config_file = os.path.join(FILE_PATH, "../", "logging-config.yaml")

    with open(config_file) as stream:
        try:
            conf = yaml.safe_load(stream)
            if safe_get(conf, ["handlers", "file", "filename"]):
                conf["handlers"]["file"]["filename"] = os.path.join(
                    os.environ.get("LOG_DIR", "."), conf["handlers"]["file"]["filename"]
                )

            # Remove all existing handlers before reconfiguring
            root_logger = logging.getLogger()
            if root_logger.hasHandlers():
                root_logger.handlers.clear()  # Remove all existing handlers

            logging.config.dictConfig(conf)

        except yaml.YAMLError as exc:
            logging.error(exc)