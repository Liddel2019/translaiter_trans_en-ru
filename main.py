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
        """
        self.config_manager = None
        self.config = {}
        self.app = None
        self.gui = None
        self.tensorboard_process = None
        self.start_time = datetime.now()
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
            project_root = self.config_manager.get_config_value("general.project_root", self.config)
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
            # Fetch logger configuration with default fallback
            log_config = self.config_manager.get_config_value("logger", self.config, default={})
            if not log_config:
                logger.warning("Logger configuration not found, using defaults")
                log_config = {
                    "log_file": "logs/translaiter.log",
                    "log_level": "INFO",
                    "max_log_size": 1048576
                }

            log_file = self.config_manager.get_absolute_path(log_config.get("log_file", "logs/translaiter.log"))
            log_level = getattr(logging, log_config.get("log_level", "INFO").upper(), logging.INFO)
            max_log_size = log_config.get("max_log_size", 1048576)

            # Create log directory
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Configure file handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_log_size,
                backupCount=5
            )
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))

            # Update logger configuration
            logger.handlers = [file_handler, logging.StreamHandler(sys.stdout)]
            logger.setLevel(log_level)
            logger.info(f"Logging configured with level {log_config.get('log_level', 'INFO')} and file {log_file}")
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
            logger.error(f"Failed to import TranslationGUI: {str(e)}")
            raise RuntimeError(f"Cannot launch GUI due to import error: {str(e)}")

        try:
            from PyQt5.QtWidgets import QApplication
            self.app = QApplication(sys.argv)
            gui_config = self.config_manager.get_config_value("gui", self.config, default={})
            window_size = gui_config.get("window_size", [400, 300])
            start_position = gui_config.get("start_position", [100, 100])

            self.gui = TranslationGUI(self.config)
            self.gui.resize(*window_size)
            self.gui.move(*start_position)
            self.gui.show()

            logger.info("GUI application launched successfully")
            sys.exit(self.app.exec_())
        except Exception as e:
            logger.error(f"Failed to launch GUI application: {str(e)}")
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
                logger.info("TensorBoard process terminated")

            runtime = datetime.now() - self.start_time
            logger.info(f"Application ran for {runtime.total_seconds()} seconds")

            if self.gui:
                self.gui.close()
                logger.info("GUI closed")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

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
            logger.info(f"Current project version: {project_version}")

            if platform.system() not in ["Windows", "Linux", "Darwin"]:
                logger.warning(f"Unsupported operating system: {platform.system()}")
                return False

            logger.info("Version and compatibility check passed")
            return True
        except Exception as e:
            logger.error(f"Update check failed: {str(e)}")
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
                self.config_manager.get_config_value("general.tensorboard_path", self.config, default="logs/tensorboard")
            )
            os.makedirs(tensorboard_path, exist_ok=True)

            try:
                self.tensorboard_process = subprocess.Popen(
                    ["tensorboard", f"--logdir={tensorboard_path}", "--port=6006"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                logger.info("TensorBoard started on port 6006")
            except FileNotFoundError:
                logger.warning("TensorBoard not installed or not found in PATH")
            except Exception as e:
                logger.error(f"Failed to start TensorBoard: {str(e)}")
        except Exception as e:
            logger.error(f"Service initialization failed: {str(e)}")
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
            logger.info("All required imports validated successfully")
            return True
        except ImportError as e:
            logger.error(f"Import validation failed: {str(e)}")
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
            initialize_project_directories(self.config)
            if not validate_config_paths(self.config):
                raise RuntimeError("Configuration path validation failed")
            logger.info("Environment setup completed successfully")
        except Exception as e:
            logger.error(f"Environment setup failed: {str(e)}")
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

        if not app_manager.validate_environment():
            logger.error("Environment validation failed. Exiting.")
            sys.exit(1)

        app_manager.setup_logging()

        if not app_manager.check_updates():
            logger.warning("Update check failed. Proceeding with caution.")

        if not app_manager.validate_imports():
            logger.error("Required imports are missing. Exiting.")
            sys.exit(1)

        app_manager.setup_environment()
        app_manager.initialize_services()
        app_manager.run_application()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        app_manager.cleanup()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
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

def initialize_project_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary project directories based on configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Raises:
        RuntimeError: If directory creation fails.

    Example:
        >>> config = {"general": {"project_root": "./"}, ...}
        >>> initialize_project_directories(config)
    """
    try:
        paths = [
            config["general"]["tensorboard_path"],
            config["dataset"]["dataset_path"],
            config["dataset"]["cache_path"],
            config["tokenizer"]["tokenizer_cache_path"],
            config["model"]["checkpoint_path"],
            config["logger"]["log_file"],
            config["metrics"]["plot_path"],
            config["unit"]["heatmap_path"],
            config["unit"]["backup_path"]
        ]
        for path in paths:
            full_path = os.path.join(config["general"]["project_root"], path)
            os.makedirs(full_path, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")
        logger.info("Project directories initialized")
    except Exception as e:
        logger.error(f"Failed to initialize project directories: {str(e)}")
        raise RuntimeError(f"Directory initialization failed: {str(e)}")

def validate_config_paths(config: Dict[str, Any]) -> bool:
    """
    Validate that all configured paths are accessible.

    Args:
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        bool: True if all paths are valid, False otherwise.

    Example:
        >>> config = {"general": {"project_root": "./"}, ...}
        >>> validate_config_paths(config)
    """
    try:
        paths = [
            config["general"]["tensorboard_path"],
            config["dataset"]["dataset_path"],
            config["dataset"]["cache_path"],
            config["tokenizer"]["tokenizer_cache_path"],
            config["model"]["checkpoint_path"],
            os.path.dirname(config["logger"]["log_file"]),
            config["metrics"]["plot_path"],
            config["unit"]["heatmap_path"],
            config["unit"]["backup_path"]
        ]
        for path in paths:
            full_path = os.path.join(config["general"]["project_root"], path)
            if not os.access(os.path.dirname(full_path) or ".", os.W_OK):
                logger.error(f"Path not writable: {full_path}")
                return False
        logger.info("All configured paths validated successfully")
        return True
    except Exception as e:
        logger.error(f"Path validation failed: {str(e)}")
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