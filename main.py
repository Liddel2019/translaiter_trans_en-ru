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

TranslationGUI = None

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ApplicationManager:
    def __init__(self):
        self.config_manager = None
        self.config = {}
        self.app = None
        self.gui = None
        self.tensorboard_process = None
        self.start_time = datetime.now()
        self.logger_instance = None

    def load_initial_config(self) -> Dict[str, Any]:
        try:
            self.config_manager = ConfigManager(config_path="config.yaml")
            self.config = self.config_manager.config
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
        try:
            required_python = (3, 8)
            current_python = sys.version_info[:2]
            if current_python < required_python:
                logger.error(f"Python version {current_python} is too old. Required: {required_python}")
                return False

            required_packages = ["PyQt5", "yaml", "psutil"]
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError:
                    logger.error(f"Required package {package} is not installed")
                    return False

            memory = psutil.virtual_memory()
            if memory.available < 2 * 1024 * 1024 * 1024:
                logger.warning("Low system memory detected: <2GB available")
                return False

            project_root = self.config_manager.get_config_value("general.project_root")
            if not os.path.exists(project_root):
                logger.error(f"Project root directory does not exist: {project_root}")
                return False

            return True
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            raise RuntimeError(f"Environment validation failed: {str(e)}")

    def setup_logging(self) -> None:
        try:
            self.logger_instance = Logger(self.config)
            if not self.logger_instance.validate_logging_setup():
                raise RuntimeError("Logging setup validation failed")
        except Exception as e:
            logger.error(f"Failed to configure logging: {str(e)}")
            raise RuntimeError(f"Logging configuration failed: {str(e)}")

    def run_application(self) -> None:
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
            sys.exit(self.app.exec_())
        except Exception as e:
            self.logger_instance.log_exception(e, "Failed to launch GUI application")
            raise RuntimeError(f"GUI application launch failed: {str(e)}")

    def cleanup(self) -> None:
        try:
            if self.tensorboard_process:
                self.tensorboard_process.terminate()
                self.tensorboard_process.wait(timeout=5)

            if self.gui:
                self.gui.close()
        except Exception as e:
            self.logger_instance.log_exception(e, "Cleanup failed")

    def check_updates(self) -> bool:
        try:
            project_version = "1.0.0"
            if platform.system() not in ["Windows", "Linux", "Darwin"]:
                self.logger_instance.log_message("WARNING", f"Unsupported operating system: {platform.system()}")
                return False
            return True
        except Exception as e:
            self.logger_instance.log_exception(e, "Update check failed")
            return False

    def initialize_services(self) -> None:
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
            except FileNotFoundError:
                self.logger_instance.log_message("WARNING", "TensorBoard not installed or not found in PATH")
            except Exception as e:
                self.logger_instance.log_exception(e, "Failed to start TensorBoard")
        except Exception as e:
            self.logger_instance.log_exception(e, "Service initialization failed")
            raise RuntimeError(f"Service initialization failed: {str(e)}")

    def parse_arguments(self) -> argparse.Namespace:
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
        try:
            global TranslationGUI
            from gui import TranslationGUI
            from utils.config import ConfigManager
            from PyQt5.QtWidgets import QApplication
            return True
        except ImportError as e:
            self.logger_instance.log_exception(e, "Import validation failed")
            return False

    def setup_environment(self) -> None:
        try:
            initialize_project_directories(self.config, self.logger_instance)
            if not validate_config_paths(self.config, self.logger_instance):
                raise RuntimeError("Configuration path validation failed")
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
    try:
        app_manager = ApplicationManager()
        args = app_manager.parse_arguments()

        if args.debug:
            logger.setLevel(logging.DEBUG)

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
    try:
        disk = psutil.disk_usage(path)
        if disk.free < 1 * 1024 * 1024 * 1024:
            return False
        return True
    except Exception as e:
        logger.error(f"Disk space check failed: {str(e)}")
        return False

def initialize_project_directories(config: Dict[str, Any], logger_instance: Logger) -> None:
    try:
        assert isinstance(config, dict), "Configuration must be a dictionary"
        assert isinstance(logger_instance, Logger), "Logger instance must be of type Logger"

        path_configs = [
            ("general.tensorboard_path", "logs/tensorboard"),
            ("dataset.dataset_path", "data/datasets"),
            ("dataset.cache_path", "data/cache"),
            ("tokenizer.tokenizer_cache_path", "data/tokenizer_cache"),
            ("model.checkpoint_path", "model/checkpoints"),
            ("metrics.plot_path", "logs/plots"),
            ("unit.heatmap_path", "logs/heatmaps"),
            ("unit.backup_path", "logs/backups")
        ]

        paths = []
        for key, default in path_configs:
            try:
                section, subkey = key.split(".")
                value = config.get(section, {}).get(subkey, default)
                if not isinstance(value, str):
                    value = default
                paths.append(value)
            except KeyError:
                paths.append(default)

        project_root = config.get("general", {}).get("project_root", "./")
        log_file = config.get("logger", {}).get("log_file", "logs/translaiter.log")
        log_dir = os.path.dirname(log_file) or "logs"

        for path in paths + [log_dir]:
            full_path = os.path.join(project_root, path)
            os.makedirs(full_path, exist_ok=True)
            logger_instance.log_message("DEBUG", f"Ensured directory exists: {full_path}")

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
    try:
        path_configs = [
            ("general.tensorboard_path", "logs/tensorboard"),
            ("dataset.dataset_path", "data/datasets"),
            ("dataset.cache_path", "data/cache"),
            ("tokenizer.tokenizer_cache_path", "data/tokenizer_cache"),
            ("model.checkpoint_path", "model/checkpoints"),
            ("metrics.plot_path", "logs/plots"),
            ("unit.heatmap_path", "logs/heatmaps"),
            ("unit.backup_path", "logs/backups")
        ]

        project_root = config.get("general", {}).get("project_root", "./")
        log_file = config.get("logger", {}).get("log_file", "logs/translaiter.log")
        log_dir = os.path.dirname(log_file) or "logs"

        for key, default in path_configs:
            try:
                section, subkey = key.split(".")
                path = config.get(section, {}).get(subkey, default)
                if not isinstance(path, str):
                    path = default
                full_path = os.path.join(project_root, path)
                if not os.access(os.path.dirname(full_path) or ".", os.W_OK):
                    logger_instance.log_message("ERROR", f"Write access denied for directory: {os.path.dirname(full_path)}")
                    return False
            except KeyError:
                full_path = os.path.join(project_root, default)
                if not os.access(os.path.dirname(full_path) or ".", os.W_OK):
                    logger_instance.log_message("ERROR", f"Write access denied for directory: {os.path.dirname(full_path)}")
                    return False

        full_log_dir = os.path.join(project_root, log_dir)
        if not os.access(full_log_dir or ".", os.W_OK):
            logger_instance.log_message("ERROR", f"Write access denied for log directory: {full_log_dir}")
            return False

        logger_instance.log_message("INFO", "All configuration paths validated successfully")
        return True
    except Exception as e:
        logger_instance.log_message("ERROR", f"Path validation failed: {str(e)}")
        return False

def log_system_info() -> None:
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
    try:
        import socket
        socket.create_connection(("www.google.com", 80), timeout=2)
        return True
    except Exception as e:
        logger.warning(f"Network connectivity check failed: {str(e)}")
        return False

def save_application_state(config: Dict[str, Any]) -> None:
    try:
        backup_path = os.path.join(
            config["general"]["project_root"],
            config["unit"]["backup_path"],
            f"app_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        )
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(backup_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        logger.info(f"Application state saved to: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to save application state: {str(e)}")

def monitor_system_resources() -> Dict[str, float]:
    try:
        metrics = {}
        metrics["cpu_percent"] = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        metrics["memory_percent"] = memory.percent
        disk = psutil.disk_usage('/')
        metrics["disk_percent"] = disk.percent
        return metrics
    except Exception as e:
        logger.error(f"Failed to monitor system resources: {str(e)}")
        return {}

if __name__ == "__main__":
    try:
        log_system_info()
        main()
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
        sys.exit(1)