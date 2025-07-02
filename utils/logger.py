import os
import logging
import logging.handlers
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import sys
import colorama
from colorama import Fore, Style
from utils.config import ConfigManager

colorama.init()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ColoredFormatter(logging.Formatter):
    def __init__(self, colors: Dict[str, str], fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        super().__init__(fmt)
        self.colors = colors
        self.color_map = {}
        try:
            for level, color in colors.items():
                if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                    logger.error(f"Invalid log level in colors: {level}")
                    raise ValueError(f"Invalid log level in colors: {level}")
                if not hasattr(Fore, color.upper()):
                    logger.error(f"Invalid color for {level}: {color}")
                    raise ValueError(f"Invalid color for {level}: {color}")
                self.color_map[level] = getattr(Fore, color.upper(), Fore.WHITE)
            logger.debug("ColoredFormatter initialized with color map")
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to initialize ColoredFormatter: {str(e)}")
            raise ValueError(f"ColoredFormatter initialization failed: {str(e)}")

    def format(self, record: logging.LogRecord):
        try:
            color = self.color_map.get(record.levelname, Fore.WHITE)
            message = super().format(record)
            return f"{color}{message}{Style.RESET_ALL}"
        except Exception as e:
            logger.error(f"Failed to format log record: {str(e)}")
            return super().format(record)

class Logger:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config_manager = ConfigManager()
        self.config = config if config is not None else self.config_manager.config
        self.logger = logging.getLogger("translaiter")
        self.log_file = None
        self.max_log_size = None
        self.colors = None
        self.detail_levels = None
        self.current_detail_level = "full"
        self.file_handler = None
        self.stream_handler = None

        try:
            self.log_file = self.config_manager.get_config_value(
                "logger.log_file", default="logs/translaiter.log"
            )
            logger.debug(f"Loaded log_file: {self.log_file}")
            self.validate_config_value("log_file", self.log_file, str, non_empty=True)

            self.max_log_size = self.config_manager.get_config_value(
                "logger.max_log_size", default=1048576
            )
            logger.debug(f"Loaded max_log_size: {self.max_log_size}")
            self.validate_config_value("max_log_size", self.max_log_size, int, positive=True)

            self.colors = self.config_manager.get_config_value(
                "logger.colors", default={
                    "DEBUG": "green", "INFO": "blue", "WARNING": "yellow",
                    "ERROR": "red", "CRITICAL": "red"
                }
            )
            logger.debug(f"Loaded colors: {self.colors}")
            self.validate_config_value("colors", self.colors, dict, non_empty=True)

            self.detail_levels = self.config_manager.get_config_value(
                "logger.detail_levels", default=["full", "minimal"]
            )
            logger.debug(f"Loaded detail_levels: {self.detail_levels}")
            self.validate_config_value("detail_levels", self.detail_levels, list, non_empty=True)

            self.log_level = self.config_manager.get_config_value(
                "general.log_level", default="INFO"
            )
            logger.debug(f"Loaded log_level: {self.log_level}")
            self.validate_config_value("log_level", self.log_level, str)
            if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.error(f"Invalid log_level: {self.log_level}")
                raise ValueError(f"Invalid log_level: {self.log_level}")

            self.validate_log_config()
            logger.info("Logging configuration validated successfully")
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Failed to initialize logger configuration: {str(e)}")
            raise ValueError(f"Logger configuration initialization failed: {str(e)}")

        try:
            self.initialize_logger()
            logger.info(f"Logger initialized with file: {self.log_file}")
        except Exception as e:
            logger.error(f"Logger initialization failed: {str(e)}")
            raise ValueError(f"Logger initialization failed: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type,
                            non_empty: bool = False, positive: bool = False):
        try:
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                raise ValueError(f"Invalid type for {key}")
            if non_empty and isinstance(value, (str, list)) and not value:
                logger.error(f"{key} cannot be empty")
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                logger.error(f"{key} must be positive")
                raise ValueError(f"{key} must be positive")
            logger.debug(f"Validated {key}: {value}")
        except (ValueError, TypeError) as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

    def validate_log_config(self):
        try:
            self.validate_config_value("log_file", self.log_file, str, non_empty=True)
            self.validate_config_value("max_log_size", self.max_log_size, int, positive=True)
            self.validate_config_value("colors", self.colors, dict, non_empty=True)
            self.validate_config_value("detail_levels", self.detail_levels, list, non_empty=True)
            self.validate_config_value("log_level", self.log_level, str)
            for level in self.detail_levels:
                if level not in ["full", "minimal"]:
                    logger.error(f"Invalid detail level: {level}")
                    raise ValueError(f"Invalid detail level: {level}")
            for level, color in self.colors.items():
                if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                    logger.error(f"Invalid log level in colors: {level}")
                    raise ValueError(f"Invalid log level in colors: {level}")
                if not hasattr(Fore, color.upper()):
                    logger.error(f"Invalid color for {level}: {color}")
                    raise ValueError(f"Invalid color for {level}: {color}")
            logger.info("Logging configuration validated successfully")
        except (ValueError, AttributeError) as e:
            logger.error(f"Logging configuration validation failed: {str(e)}")
            raise ValueError(f"Logging configuration validation failed: {str(e)}")

    def initialize_logger(self):
        try:
            self.logger.handlers.clear()
            logger.debug("Cleared existing logger handlers")

            self.logger.setLevel(getattr(logging, self.log_level, logging.INFO))
            logger.debug(f"Set logger level to: {self.log_level}")

            log_file_path = self.config_manager.get_absolute_path(self.log_file)
            try:
                os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                logger.debug(f"Ensured log directory exists: {os.path.dirname(log_file_path)}")
            except (OSError, IOError) as e:
                logger.error(f"Failed to create log directory: {str(e)}")
                raise ValueError(f"Log directory creation failed: {str(e)}")

            self.file_handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=self.max_log_size,
                backupCount=5
            )
            self.file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            self.file_handler.setFormatter(file_formatter)
            self.logger.addHandler(self.file_handler)
            logger.info(f"File handler configured with path: {log_file_path}, max size: {self.max_log_size}")

            self.stream_handler = logging.StreamHandler(sys.stdout)
            self.stream_handler.setLevel(logging.DEBUG)
            stream_formatter = ColoredFormatter(self.colors)
            self.stream_handler.setFormatter(stream_formatter)
            self.logger.addHandler(self.stream_handler)
            logger.info("Stream handler configured with colored output")

            logger.info("Logging handlers configured successfully")
        except (ValueError, IOError, OSError) as e:
            logger.error(f"Failed to initialize logger: {str(e)}")
            raise ValueError(f"Logger initialization failed: {str(e)}")

    def validate_logging_setup(self):
        try:
            if not self.file_handler or not self.stream_handler:
                logger.error("File or stream handler not initialized")
                return False
            log_file_path = self.config_manager.get_absolute_path(self.log_file)
            if not os.path.exists(os.path.dirname(log_file_path)):
                logger.error("Log file directory does not exist")
                return False
            if self.get_log_file_size() > self.max_log_size:
                logger.warning("Log file size exceeds max_log_size, consider rotating")
            logger.info("Logging setup validated successfully")
            return True
        except (OSError, ValueError) as e:
            logger.error(f"Logging setup validation failed: {str(e)}")
            return False

    def customize_formatter(self, fmt: Optional[str] = None):
        try:
            if fmt is None:
                fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            if not isinstance(fmt, str):
                logger.error(f"Invalid formatter format: {fmt}")
                raise ValueError(f"Invalid formatter format: {fmt}")
            if self.file_handler:
                self.file_handler.setFormatter(logging.Formatter(fmt))
                logger.debug("Updated file handler formatter")
            if self.stream_handler:
                self.stream_handler.setFormatter(ColoredFormatter(self.colors, fmt))
                logger.debug("Updated stream handler formatter")
            logger.info(f"Log formatter customized to: {fmt}")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to customize formatter: {str(e)}")
            raise ValueError(f"Formatter customization failed: {str(e)}")

    def set_detail_level(self, level: str):
        try:
            if not isinstance(level, str):
                logger.error(f"Invalid detail level type: {type(level)}")
                raise ValueError(f"Invalid detail level type: {type(level)}")
            if level not in self.detail_levels:
                logger.error(f"Invalid detail level: {level}, available: {self.detail_levels}")
                raise ValueError(f"Invalid detail level: {level}")
            self.current_detail_level = level
            if level == "minimal":
                self.logger.setLevel(logging.WARNING)
                if self.file_handler:
                    self.file_handler.setLevel(logging.WARNING)
                if self.stream_handler:
                    self.stream_handler.setLevel(logging.WARNING)
            else:
                self.logger.setLevel(logging.DEBUG)
                if self.file_handler:
                    self.file_handler.setLevel(logging.DEBUG)
                if self.stream_handler:
                    self.stream_handler.setLevel(logging.DEBUG)
            logger.info(f"Set logging detail level to: {level}")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to set detail level: {str(e)}")
            raise ValueError(f"Failed to set detail level: {str(e)}")

    def log_message(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        try:
            if not isinstance(level, str):
                logger.error(f"Invalid log level type: {type(level)}")
                raise ValueError(f"Invalid log level type: {type(level)}")
            if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.error(f"Invalid log level: {level}")
                raise ValueError(f"Invalid log level: {level}")
            if not isinstance(message, str):
                logger.error(f"Invalid message type: {type(message)}")
                raise ValueError(f"Invalid message type: {type(message)}")
            log_func = getattr(self.logger, level.lower())
            if self.current_detail_level == "minimal" and level in ["DEBUG", "INFO"]:
                return
            log_func(message, extra=extra)
            logger.debug(f"Logged message at level {level}: {message}")
        except (ValueError, AttributeError) as e:
            logger.error(f"Failed to log message: {str(e)}")
            raise ValueError(f"Failed to log message: {str(e)}")

    def rotate_logs(self):
        try:
            if self.file_handler:
                self.file_handler.doRollover()
                logger.info("Log file rotation triggered successfully")
            else:
                logger.warning("No file handler available for rotation")
        except (IOError, OSError) as e:
            logger.error(f"Failed to rotate logs: {str(e)}")
            raise ValueError(f"Log rotation failed: {str(e)}")

    def clear_handlers(self):
        try:
            self.logger.handlers.clear()
            self.file_handler = None
            self.stream_handler = None
            logger.info("All logger handlers cleared")
        except Exception as e:
            logger.error(f"Failed to clear handlers: {str(e)}")
            raise ValueError(f"Failed to clear handlers: {str(e)}")

    def get_log_file_size(self):
        try:
            log_file_path = self.config_manager.get_absolute_path(self.log_file)
            if os.path.exists(log_file_path):
                size = os.path.getsize(log_file_path)
                logger.debug(f"Log file size: {size} bytes")
                return size
            logger.warning(f"Log file does not exist: {log_file_path}")
            return 0
        except (OSError, IOError) as e:
            logger.error(f"Failed to get log file size: {str(e)}")
            raise ValueError(f"Failed to get log file size: {str(e)}")

    def archive_logs(self, archive_path: str):
        try:
            if not isinstance(archive_path, str):
                logger.error(f"Invalid archive_path type: {type(archive_path)}")
                raise ValueError(f"Invalid archive_path type: {type(archive_path)}")
            log_file_path = self.config_manager.get_absolute_path(self.log_file)
            archive_path = self.config_manager.get_absolute_path(archive_path)
            os.makedirs(os.path.dirname(archive_path), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_file = f"{archive_path}_{timestamp}.log"
            if os.path.exists(log_file_path):
                with open(log_file_path, 'r') as src, open(archive_file, 'w') as dst:
                    dst.write(src.read())
                logger.info(f"Log file archived to: {archive_file}")
                self.rotate_logs()
            else:
                logger.warning(f"No log file to archive: {log_file_path}")
        except (IOError, OSError, ValueError) as e:
            logger.error(f"Failed to archive logs: {str(e)}")
            raise ValueError(f"Log archiving failed: {str(e)}")

    def log_config(self):
        try:
            logger.info("Current logging configuration:")
            logger.info(f"  Log File: {self.log_file}")
            logger.info(f"  Max Log Size: {self.max_log_size} bytes")
            logger.info(f"  Colors: {self.colors}")
            logger.info(f"  Detail Levels: {self.detail_levels}")
            logger.info(f"  Current Detail Level: {self.current_detail_level}")
            logger.info(f"  Log Level: {self.log_level}")
            logger.info(f"  Log File Size: {self.get_log_file_size()} bytes")
        except Exception as e:
            logger.error(f"Failed to log configuration: {str(e)}")
            raise ValueError(f"Failed to log configuration: {str(e)}")

    def shutdown_logger(self):
        try:
            self.clear_handlers()
            logging.shutdown()
            logger.info("Logger shutdown successfully")
        except Exception as e:
            logger.error(f"Failed to shutdown logger: {str(e)}")
            raise ValueError(f"Failed to shutdown logger: {str(e)}")

    def log_metrics(self, metrics: Dict[str, Any], level: str = "INFO"):
        try:
            if not isinstance(metrics, dict):
                logger.error(f"Invalid metrics type: {type(metrics)}")
                raise ValueError(f"Invalid metrics type: {type(metrics)}")
            metrics_str = ", ".join(f"{k}: {v}" for k, v in metrics.items())
            self.log_message(level, f"Metrics: {metrics_str}")
            logger.debug(f"Logged metrics: {metrics_str}")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            raise ValueError(f"Failed to log metrics: {str(e)}")

    def log_exception(self, exc: Exception, message: str = "An exception occurred"):
        try:
            if not isinstance(exc, Exception):
                logger.error(f"Invalid exception type: {type(exc)}")
                raise ValueError(f"Invalid exception type: {type(exc)}")
            if not isinstance(message, str):
                logger.error(f"Invalid message type: {type(message)}")
                raise ValueError(f"Invalid message type: {type(message)}")
            self.logger.exception(f"{message}: {str(exc)}")
            logger.debug(f"Logged exception: {str(exc)}")
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to log exception: {str(e)}")
            raise ValueError(f"Failed to log exception: {str(e)}")

    def check_log_file_health(self):
        try:
            log_file_path = self.config_manager.get_absolute_path(self.log_file)
            if not os.path.exists(os.path.dirname(log_file_path)):
                logger.error(f"Log file directory does not exist: {os.path.dirname(log_file_path)}")
                return False
            if os.path.exists(log_file_path):
                with open(log_file_path, 'a') as f:
                    pass
            logger.debug("Log file health check passed")
            return True
        except (IOError, OSError) as e:
            logger.error(f"Log file health check failed: {str(e)}")
            return False

    def flush_logs(self):
        try:
            for handler in self.logger.handlers:
                handler.flush()
            logger.info("Logs flushed successfully")
        except (IOError, OSError) as e:
            logger.error(f"Failed to flush logs: {str(e)}")
            raise ValueError(f"Failed to flush logs: {str(e)}")

if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        logger_instance = Logger(config)

        logger_instance.log_config()

        logger_instance.log_message("DEBUG", "This is a debug message", extra={"context": "test"})
        logger_instance.log_message("INFO", "This is an info message", extra={"context": "test"})
        logger_instance.log_message("WARNING", "This is a warning message", extra={"context": "test"})
        logger_instance.log_message("ERROR", "This is an error message", extra={"context": "test"})
        logger_instance.log_message("CRITICAL", "This is a critical message", extra={"context": "test"})

        logger_instance.set_detail_level("minimal")
        logger_instance.log_message("DEBUG", "This debug message should not appear in minimal mode")
        logger_instance.log_message("WARNING", "This warning message should appear")
        logger_instance.set_detail_level("full")
        logger_instance.log_message("DEBUG", "This debug message should appear in full mode")

        sample_metrics = {"epoch": 1, "loss": 0.5, "learning_rate": 0.0001}
        logger_instance.log_metrics(sample_metrics, level="INFO")

        try:
            raise ValueError("Test exception for logging")
        except ValueError as e:
            logger_instance.log_exception(e, "Caught test exception")

        logger_instance.rotate_logs()

        is_healthy = logger_instance.check_log_file_health()
        logger_instance.log_message("INFO", f"Log file health check: {'Healthy' if is_healthy else 'Unhealthy'}")

        logger_instance.archive_logs("logs/archives/translaiter_archive.log")

        logger_instance.customize_formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        logger_instance.log_message("INFO", "Test message with custom formatter")

        is_valid = logger_instance.validate_logging_setup()
        logger_instance.log_message("INFO", f"Logging setup validation: {'Valid' if is_valid else 'Invalid'}")

        logger_instance.flush_logs()

        logger_instance.shutdown_logger()

        logger.info("Logger test completed successfully")
    except Exception as e:
        logger.error(f"Logger test execution failed: {str(e)}")