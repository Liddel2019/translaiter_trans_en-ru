# main.py
import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import subprocess
import platform
import psutil
import yaml
from utils.config import ConfigManager
from utils.logger import Logger

# Lazy import to avoid circular dependencies
TranslationGUI = None

# Configure initial logging before config-based logging is set up
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ApplicationManager:
    """Manages the initialization and lifecycle of the translaiter_trans_en-ru application."""

    def __init__(self):
        """
        Initialize the ApplicationManager with default state.

        Attributes:
            config_manager (ConfigManager): Manages configuration settings.
            config (Dict[str, Any]): Loaded configuration dictionary.
            app (QApplication): PyQt5 application instance.
            gui (TranslationGUI): GUI instance for the application.
            tensorboard_process (subprocess.Popen): TensorBoard process, if running.
            start_time (datetime): Application start time for runtime tracking.
            logger_instance (Logger): Custom logger instance.
        """
        self.config_manager = None
        self.config = {}
        self.app = None
        self.gui = None
        self.tensorboard_process = None
        self.start_time = datetime.now()
        self.logger_instance = None
        logger.info("ApplicationManager initialized")

    def load_initial_config(self) -> Dict[str, Any]:
        """
        Load the initial configuration from config.yaml using ConfigManager.

        Returns:
            Dict[str, Any]: Loaded configuration dictionary.

        Raises:
            FileNotFoundError: If config.yaml is missing.
            yaml.YAMLError: If config.yaml has invalid syntax.
            RuntimeError: For other configuration loading errors.

        Example:
            >>> app_manager = ApplicationManager()
            >>> config = app_manager.load_initial_config()
        """
        try:
            self.config_manager = ConfigManager(config_path="config.yaml")
            self.config = self.config_manager.config
            logger.info("Initial configuration loaded successfully via ConfigManager")
            return self.config
        except FileNotFoundError:
            logger.error("config.yaml not found in project root")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML syntax in config.yaml: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {str(e)}")
            raise RuntimeError(f"Configuration loading failed: {str(e)}")

    def validate_environment(self) -> bool:
        """
        Validate the runtime environment for required dependencies and resources.

        Returns:
            bool: True if environment is valid, False otherwise.

        Raises:
            RuntimeError: If environment validation fails critically.

        Example:
            >>> app_manager = ApplicationManager()
            >>> is_valid = app_manager.validate_environment()
        """
        try:
            # Check Python version
            required_python = (3, 8)
            current_python = sys.version_info[:2]
            if current_python < required_python:
                logger.error(f"Python version {current_python} is too old. Required: {required_python}")
                return False

            # Check required packages
            required_packages = ["PyQt5", "yaml", "psutil"]
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    logger.error(f"Required package {package} is not installed")
                    return False

            # Check system resources
            memory = psutil.virtual_memory()
            if memory.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB available
                logger.warning("Low system memory detected: <2GB available")
                return False

            # Check project root existence
            project_root = self.config_manager.get_config_value("general.project_root")
            if not os.path.exists(project_root):
                logger.error(f"Project root directory does not exist: {project_root}")
                return False

            logger.info("Environment validation successful")
            return True
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            raise RuntimeError(f"Environment validation failed: {str(e)}")

    def setup_logging(self) -> None:
        """
        Configure logging based on settings from config.yaml via ConfigManager.

        Raises:
            RuntimeError: If logging configuration fails.

        Example:
            >>> app_manager = ApplicationManager()
            >>> app_manager.load_initial_config()
            >>> app_manager.setup_logging()
        """
        try:
            self.logger_instance = Logger(self.config)
            if not self.logger_instance.validate_logging_setup():
                raise RuntimeError("Logging setup validation failed")
            self.logger_instance.log_message("INFO", "Logging configured successfully via Logger class")
        except Exception as e:
            logger.error(f"Failed to configure logging: {str(e)}")
            raise RuntimeError(f"Logging configuration failed: {str(e)}")

    def run_application(self) -> None:
        """
        Launch the PyQt5 GUI application with configuration settings.

        Raises:
            RuntimeError: If GUI application fails to launch or TranslationGUI import fails.

        Example:
            >>> app_manager = ApplicationManager()
            >>> app_manager.run_application()
        """
        global TranslationGUI
        try:
            from gui import TranslationGUI
        except ImportError as e:
            self.logger_instance.log_exception(e, "Failed to import TranslationGUI")
            raise RuntimeError(f"Cannot launch GUI due to import error: {str(e)}")

        try:
            from PyQt5.QtWidgets import QApplication
            self.app = QApplication(sys.argv)
            window_size = self.config_manager.get_config_value("gui.window_size", default=[400, 300])
            start_position = self.config_manager.get_config_value("gui.start_position", default=[100, 100])

            self.gui = TranslationGUI(self.config)
            self.gui.resize(*window_size)
            self.gui.move(*start_position)
            self.gui.show()

            self.logger_instance.log_message("INFO", "GUI application launched successfully")
            sys.exit(self.app.exec_())
        except Exception as e:
            self.logger_instance.log_exception(e, "Failed to launch GUI application")
            raise RuntimeError(f"GUI application launch failed: {str(e)}")

    def cleanup(self) -> None:
        """
        Clean up resources on application exit.

        Example:
            >>> app_manager = ApplicationManager()
            >>> app_manager.cleanup()
        """
        try:
            if self.tensorboard_process:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)
                self.logger_instance.log_message("INFO", "TensorBoard process terminated")

            runtime = datetime.now() - self.start_time
            self.logger_instance.log_message("INFO", f"Application ran for {runtime.total_seconds()} seconds")

            if self.gui:
                self.gui.close()
                self.logger_instance.log_message("INFO", "GUI closed")
        except Exception as e:
            self.logger_instance.log_exception(e, "Cleanup failed")

    def check_updates(self) -> bool:
        """
        Check for project updates or version compatibility.

        Returns:
            bool: True if compatible/no updates needed, False if issues detected.

        Example:
            >>> app_manager = ApplicationManager()
            >>> app_manager.check_updates()
        """
        try:
            project_version = "1.0.0"
            self.logger_instance.log_message("INFO", f"Current project version: {project_version}")

            if platform.system() not in ["Windows", "Linux", "Darwin"]:
                self.logger_instance.log_message("WARNING", f"Unsupported operating system: {platform.system()}")
                return False

            self.logger_instance.log_message("INFO", "Version and compatibility check passed")
            return True
        except Exception as e:
            self.logger_instance.log_exception(e, "Update check failed")
            return False

    def initialize_services(self) -> None:
        """
        Initialize additional services like TensorBoard using configuration settings.

        Raises:
            RuntimeError: If service initialization fails.

        Example:
            >>> app_manager = ApplicationManager()
            >>> app_manager.initialize_services()
        """
        try:
            tensorboard_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value("general.tensorboard_path", default="logs/tensorboard")
            )
            os.makedirs(tensorboard_path, exist_ok=True)

            try:
                self.tensorboard_process = subprocess.Popen(
                    ["tensorboard", f"--logdir={tensorboard_path}", "--port=6006"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.logger_instance.log_message("INFO", "TensorBoard started on port 6006")
            except FileNotFoundError:
                self.logger_instance.log_message("WARNING", "TensorBoard not installed or not found in PATH")
            except Exception as e:
                self.logger_instance.log_exception(e, "Failed to start TensorBoard")
        except Exception as e:
            self.logger_instance.log_exception(e, "Service initialization failed")
            raise RuntimeError(f"Service initialization failed: {str(e)}")

    def parse_arguments(self) -> argparse.Namespace:
        """
        Parse command-line arguments for the application.

        Returns:
            argparse.Namespace: Parsed command-line arguments.

        Example:
            >>> app_manager = ApplicationManager()
            >>> args = app_manager.parse_arguments()
        """
        parser = argparse.ArgumentParser(description="Translaiter: English to Russian Translation System")
        parser.add_argument(
            "--config",
            type=str,
            default="config.yaml",
            help="Path to the configuration file"
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Run in debug mode with verbose logging"
        )
        return parser.parse_args()

    def validate_imports(self) -> bool:
        """
        Validate that all required imports are available.

        Returns:
            bool: True if all imports are successful, False otherwise.

        Example:
            >>> app_manager = ApplicationManager()
            >>> app_manager.validate_imports()
        """
        try:
            global TranslationGUI
            from gui import TranslationGUI
            from utils.config import ConfigManager
            from PyQt5.QtWidgets import QApplication
            self.logger_instance.log_message("INFO", "All required imports validated successfully")
            return True
        except ImportError as e:
            self.logger_instance.log_exception(e, "Import validation failed")
            return False

    def setup_environment(self) -> None:
        """
        Set up the application environment, including directories and resources.

        Raises:
            RuntimeError: If environment setup fails.

        Example:
            >>> app_manager = ApplicationManager()
            >>> app_manager.setup_environment()
        """
        try:
            self.logger_instance.log_message("INFO", "Starting environment setup")
            initialize_project_directories(self.config, self.logger_instance)
            if not validate_config_paths(self.config, self.logger_instance):
                raise RuntimeError("Configuration path validation failed")
            self.logger_instance.log_message("INFO", "Environment setup completed successfully")
        except KeyError as e:
            self.logger_instance.log_exception(e, "Configuration key error during environment setup")
            raise RuntimeError(f"Environment setup failed: Missing configuration key {str(e)}")
        except TypeError as e:
            self.logger_instance.log_exception(e, "Type error during environment setup")
            raise RuntimeError(f"Environment setup failed: Invalid type {str(e)}")
        except OSError as e:
            self.logger_instance.log_exception(e, "OS error during environment setup")
            raise RuntimeError(f"Environment setup failed: OS error {str(e)}")
        except Exception as e:
            self.logger_instance.log_exception(e, "Unexpected error during environment setup")
            raise RuntimeError(f"Environment setup failed: {str(e)}")

def main() -> None:
    """
    Main entry point for the translaiter_trans_en-ru application.

    Example:
        >>> main()
    """
    try:
        app_manager = ApplicationManager()
        args = app_manager.parse_arguments()

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        app_manager.load_initial_config()
        app_manager.setup_logging()

        if not app_manager.validate_environment():
            app_manager.logger_instance.log_message("ERROR", "Environment validation failed. Exiting.")
            sys.exit(1)

        if not app_manager.check_updates():
            app_manager.logger_instance.log_message("WARNING", "Update check failed. Proceeding with caution.")

        if not app_manager.validate_imports():
            app_manager.logger_instance.log_message("ERROR", "Required imports are missing. Exiting.")
            sys.exit(1)

        app_manager.setup_environment()
        app_manager.initialize_services()
        app_manager.run_application()

    except KeyboardInterrupt:
        app_manager.logger_instance.log_message("INFO", "Application interrupted by user")
        app_manager.cleanup()
        sys.exit(0)
    except Exception as e:
        app_manager.logger_instance.log_exception(e, "Application failed")
        app_manager.cleanup()
        sys.exit(1)
    finally:
        app_manager.cleanup()

def check_disk_space(path: str) -> bool:
    """
    Check if sufficient disk space is available at the given path.

    Args:
        path (str): Path to check disk space for.

    Returns:
        bool: True if sufficient space, False otherwise.

    Example:
        >>> check_disk_space("./")
    """
    try:
        disk = psutil.disk_usage(path)
        if disk.free < 1 * 1024 * 1024 * 1024:
            logger.warning(f"Low disk space at {path}: {disk.free / (1024*1024)} MB available")
            return False
        logger.info(f"Disk space check passed for {path}")
        return True
    except Exception as e:
        logger.error(f"Disk space check failed: {str(e)}")
        return False

def initialize_project_directories(config: Dict[str, Any], logger_instance: Logger) -> None:
    """
    Create necessary project directories based on configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        logger_instance (Logger): Logger instance for logging.

    Raises:
        RuntimeError: If directory creation fails or configuration is invalid.

    Example:
        >>> config = {"general": {"project_root": "./"}, ...}
        >>> initialize_project_directories(config, logger_instance)
    """
    try:
        logger_instance.log_message("INFO", "Starting project directory initialization")
        assert isinstance(config, dict), "Configuration must be a dictionary"
        assert isinstance(logger_instance, Logger), "Logger instance must be of type Logger"

        # Define required configuration keys with defaults
        path_configs = [
            ("general.tensorboard_path", "logs/tensorboard"),
            ("dataset.dataset_path", "data/datasets"),
            ("dataset.cache_path", "data/cache"),
            ("tokenizer.tokenizer_cache_path", "data/tokenizer_cache"),
            ("model.checkpoint_path", "model/checkpoints"),
            ("logger.log_file", "logs/translaiter.log"),
            ("metrics.plot_path", "logs/plots"),
            ("unit.heatmap_path", "logs/heatmaps"),
            ("unit.backup_path", "logs/backups")
        ]

        paths = []
        for key, default in path_configs:
            try:
                section, subkey = key.split(".")
                value = config.get(section, {}).get(subkey, default)
                logger_instance.log_message("DEBUG", f"Validated config key {key}: {value}")
                if not isinstance(value, str):
                    logger_instance.log_message("WARNING", f"Invalid type for {key}: expected str, got {type(value)}, using default {default}")
                    value = default
                paths.append(value)
            except KeyError as e:
                logger_instance.log_message("WARNING", f"Missing config key {key}, using default {default}")
                paths.append(default)

        project_root = config.get("general", {}).get("project_root", "./")
        logger_instance.log_message("DEBUG", f"Using project root: {project_root}")

        for path in paths:
            try:
                full_path = os.path.join(project_root, path)
                os.makedirs(full_path, exist_ok=True)
                logger_instance.log_message("DEBUG", f"Created directory: {full_path}")
            except OSError as e:
                logger_instance.log_message("ERROR", f"Failed to create directory {full_path}: {str(e)}")
                raise RuntimeError(f"Directory creation failed for {full_path}: {str(e)}")

        logger_instance.log_message("INFO", "Project directories initialized successfully")
    except AssertionError as e:
        logger_instance.log_message("ERROR", f"Invalid input: {str(e)}")
        raise RuntimeError(f"Directory initialization failed: {str(e)}")
    except KeyError as e:
        logger_instance.log_message("ERROR", f"Missing configuration key: {str(e)}")
        raise RuntimeError(f"Directory initialization failed: Missing configuration key {str(e)}")
    except TypeError as e:
        logger_instance.log_message("ERROR", f"Type error in configuration: {str(e)}")
        raise RuntimeError(f"Directory initialization failed: Invalid type {str(e)}")
    except OSError as e:
        logger_instance.log_message("ERROR", f"OS error during directory creation: {str(e)}")
        raise RuntimeError(f"Directory initialization failed: OS error {str(e)}")
    except Exception as e:
        logger_instance.log_message("ERROR", f"Unexpected error during directory initialization: {str(e)}")
        raise RuntimeError(f"Directory initialization failed: {str(e)}")

def validate_config_paths(config: Dict[str, Any], logger_instance: Logger) -> bool:
    """
    Validate that all configured paths are accessible.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        logger_instance (Logger): Logger instance for logging.

    Returns:
        bool: True if all paths are valid, False otherwise.

    Example:
        >>> config = {"general": {"project_root": "./"}, ...}
        >>> validate_config_paths(config, logger_instance)
    """
    try:
        logger_instance.log_message("INFO", "Starting configuration path validation")
        path_configs = [
            ("general.tensorboard_path", "logs/tensorboard"),
            ("dataset.dataset_path", "data/datasets"),
            ("dataset.cache_path", "data/cache"),
            ("tokenizer.tokenizer_cache_path", "data/tokenizer_cache"),
            ("model.checkpoint_path", "model/checkpoints"),
            ("logger.log_file", "logs/translaiter.log"),
            ("metrics.plot_path", "logs/plots"),
            ("unit.heatmap_path", "logs/heatmaps"),
            ("unit.backup_path", "logs/backups")
        ]

        project_root = config.get("general", {}).get("project_root", "./")
        logger_instance.log_message("DEBUG", f"Using project root: {project_root}")

        for key, default in path_configs:
            try:
                section, subkey = key.split(".")
                path = config.get(section, {}).get(subkey, default)
                logger_instance.log_message("DEBUG", f"Validated config key {key}: {path}")
                if not isinstance(path, str):
                    logger_instance.log_message("WARNING", f"Invalid type for {key}: expected str, got {type(path)}, using default {default}")
                    path = default
                full_path = os.path.join(project_root, path)
                if not os.access(os.path.dirname(full_path) or ".", os.W_OK):
                    logger_instance.log_message("ERROR", f"Path not writable: {full_path}")
                    return False
            except KeyError as e:
                logger_instance.log_message("WARNING", f"Missing config key {key}, using default {default}")
                full_path = os.path.join(project_root, default)
                if not os.access(os.path.dirname(full_path) or ".", os.W_OK):
                    logger_instance.log_message("ERROR", f"Path not writable: {full_path}")
                    return False

        logger_instance.log_message("INFO", "All configured paths validated successfully")
        return True
    except Exception as e:
        logger_instance.log_message("ERROR", f"Path validation failed: {str(e)}")
        return False

def log_system_info() -> None:
    """
    Log system information for debugging and monitoring.

    Example:
        >>> log_system_info()
    """
    try:
        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"CPU cores: {psutil.cpu_count()}")
        memory = psutil.virtual_memory()
        logger.info(f"Total memory: {memory.total / (1024*1024*1024):.2f} GB")
        logger.info(f"Available memory: {memory.available / (1024*1024*1024):.2f} GB")
    except Exception as e:
        logger.error(f"Failed to log system info: {str(e)}")

def check_network_connectivity() -> bool:
    """
    Check network connectivity for potential online resources.

    Returns:
        bool: True if connected, False otherwise.

    Example:
        >>> check_network_connectivity()
    """
    try:
        import socket
        socket.create_connection(("www.google.com", 80), timeout=2)
        logger.info("Network connectivity check passed")
        return True
    except Exception as e:
        logger.warning(f"Network connectivity check failed: {str(e)}")
        return False

def save_application_state(config: Dict[str, Any]) -> None:
    """
    Save current application state to a backup file.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Example:
        >>> config = {"general": {"project_root": "./"}, ...}
        >>> save_application_state(config)
    """
    try:
        backup_path = os.path.join(
            config["general"]["project_root"],
            config["unit"]["backup_path"],
            f"app_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        with open(backup_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logger.info(f"Application state saved to {backup_path}")
    except Exception as e:
        logger.error(f"Failed to save application state: {str(e)}")

def monitor_system_resources() -> Dict[str, float]:
    """
    Monitor system resource usage.

    Returns:
        Dict[str, float]: Dictionary of resource metrics.

    Example:
        >>> monitor_system_resources()
    """
    try:
        metrics = {}
        metrics["cpu_percent"] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        metrics["memory_percent"] = memory.percent
        disk = psutil.disk_usage('/')
        metrics["disk_percent"] = disk.percent
        logger.debug(f"System resources: CPU={metrics['cpu_percent']}%, "
                    f"Memory={metrics['memory_percent']}%, Disk={metrics['disk_percent']}%")
        return metrics
    except Exception as e:
        logger.error(f"Failed to monitor system resources: {str(e)}")
        return {}

if __name__ == "__main__":
    try:
        log_system_info()
        main()
        logger.info("Main application executed successfully")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        sys.exit(1)