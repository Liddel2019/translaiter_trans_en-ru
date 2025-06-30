import os
import yaml
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from copy import deepcopy

# Configure logging for the config module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration for the translaiter_trans_en-ru project."""

    # Default configuration template
    DEFAULT_CONFIG = {
        "general": {
            "project_root": "./",
            "log_level": "INFO",
            "tensorboard_path": "logs/tensorboard"
        },
        "dataset": {
            "dataset_path": "data/datasets",
            "available_datasets": ["OPUS Tatoeba", "TED2020", "WMT'19 en-ru", "Common Crawl"],
            "max_length": 10,
            "batch_size": 32,
            "cache_path": "data/cache",
            "clean_rules": ["<", ">", "&", "\n"]
        },
        "tokenizer": {
            "tokenizer_name": "Helsinki-NLP/opus-mt-en-ru",
            "tokenizer_cache_path": "data/tokenizer_cache",
            "max_length": 10,
            "vocab_size": 32000
        },
        "model": {
            "num_layers": 6,
            "num_heads": 8,
            "hidden_size": 512,
            "dropout_rate": 0.1,
            "checkpoint_path": "model/checkpoints"
        },
        "training": {
            "learning_rate": 0.0001,
            "epochs": 10,
            "log_steps": 100,
            "beam_size": 5,
            "sample_size": 10,
            "optimizer_type": "AdamW",
            "gradient_accumulation_steps": 1
        },
        "scheduler": {
            "step_size": 10,
            "gamma": 0.1,
            "min_lr": 0.00001
        },
        "logger": {
            "log_file": "logs/translaiter.log",
            "max_log_size": 1048576,
            "colors": {
                "DEBUG": "green",
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red"
            },
            "detail_levels": ["full", "minimal"]
        },
        "metrics": {
            "bleu_weights": [0.25, 0.25, 0.25, 0.25],
            "loss_type": "cross_entropy",
            "thresholds": {"bleu": 0.3},
            "plot_path": "logs/plots"
        },
        "unit": {
            "heatmap_size": [10, 10],
            "heatmap_path": "logs/heatmaps",
            "backup_path": "logs/backups",
            "target_size": [5, 5]
        },
        "gui": {
            "window_size": [400, 300],
            "start_position": [100, 100]
        }
    }

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize ConfigManager with a configuration file path, ensuring config.yaml is loaded from project root.

        Args:
            config_path (str): Path to the configuration file, default is 'config.yaml' in project root.
        """
        self.config_path = config_path
        self.config = {}
        # Resolve config_path relative to project_root if necessary
        self._load_config_with_fallback()

    def _load_config_with_fallback(self) -> None:
        """
        Helper method to load config.yaml with fallback to default configuration if loading fails.
        Ensures robust initialization with appropriate error handling.
        """
        try:
            # Attempt to load config.yaml from the project root
            config_path = os.path.join(self.DEFAULT_CONFIG["general"]["project_root"], self.config_path)
            self.config = self.load_config(config_path)
            logger.info("Configuration initialized successfully")
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to initialize configuration: {str(e)}")
            logger.warning("Using default configuration due to initialization failure")
            self.config = deepcopy(self.DEFAULT_CONFIG)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            config_path (str): Path to the configuration file.

        Returns:
            Dict[str, Any]: Loaded configuration dictionary.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the YAML syntax is invalid.
        """
        try:
            # Ensure the path is absolute relative to project_root
            config_path = os.path.abspath(config_path)
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file) or {}
            logger.info(f"Successfully loaded configuration from {config_path}")
            if not self.validate_config(self.config):
                logger.warning("Configuration validation failed, loading defaults")
                self.config = deepcopy(self.DEFAULT_CONFIG)
            return self.config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            self.config = deepcopy(self.DEFAULT_CONFIG)
            logger.info("Loaded default configuration")
            return self.config
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML syntax in {config_path}: {str(e)}")
            raise

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration keys and types.

        Args:
            config (Dict[str, Any]): Configuration dictionary to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        try:
            # Check if all required sections exist
            required_sections = set(self.DEFAULT_CONFIG.keys())
            if not all(section in config for section in required_sections):
                logger.error("Missing required configuration sections")
                return False

            # Validate general settings
            if not isinstance(config["general"]["project_root"], str):
                logger.error("Invalid type for project_root")
                return False
            if config["general"]["log_level"] not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
                logger.error("Invalid log_level value")
                return False

            # Validate dataset settings
            if not isinstance(config["dataset"]["available_datasets"], list):
                logger.error("available_datasets must be a list")
                return False
            if not isinstance(config["dataset"]["max_length"], int) or config["dataset"]["max_length"] <= 0:
                logger.error("max_length must be a positive integer")
                return False
            if not isinstance(config["dataset"]["batch_size"], int) or config["dataset"]["batch_size"] <= 0:
                logger.error("batch_size must be a positive integer")
                return False

            # Validate tokenizer settings
            if not isinstance(config["tokenizer"]["vocab_size"], int) or config["tokenizer"]["vocab_size"] <= 0:
                logger.error("vocab_size must be a positive integer")
                return False

            # Validate model settings
            if not isinstance(config["model"]["num_layers"], int) or config["model"]["num_layers"] <= 0:
                logger.error("num_layers must be a positive integer")
                return False
            if not isinstance(config["model"]["num_heads"], int) or config["model"]["num_heads"] <= 0:
                logger.error("num_heads must be a positive integer")
                return False
            if not isinstance(config["model"]["hidden_size"], int) or config["model"]["hidden_size"] <= 0:
                logger.error("hidden_size must be a positive integer")
                return False
            if not isinstance(config["model"]["dropout_rate"], float) or not 0 <= config["model"]["dropout_rate"] <= 1:
                logger.error("dropout_rate must be a float between 0 and 1")
                return False

            # Validate training settings
            if not isinstance(config["training"]["learning_rate"], float) or config["training"]["learning_rate"] <= 0:
                logger.error("learning_rate must be a positive float")
                return False
            if not isinstance(config["training"]["epochs"], int) or config["training"]["epochs"] <= 0:
                logger.error("epochs must be a positive integer")
                return False

            # Validate scheduler settings
            if not isinstance(config["scheduler"]["gamma"], float) or not 0 < config["scheduler"]["gamma"] <= 1:
                logger.error("gamma must be a float between 0 and 1")
                return False

            # Validate logger settings
            if not isinstance(config["logger"]["max_log_size"], int) or config["logger"]["max_log_size"] <= 0:
                logger.error("max_log_size must be a positive integer")
                return False
            if not isinstance(config["logger"]["colors"], dict):
                logger.error("colors must be a dictionary")
                return False

            # Validate metrics settings
            if not isinstance(config["metrics"]["bleu_weights"], list) or len(config["metrics"]["bleu_weights"]) != 4:
                logger.error("bleu_weights must be a list of 4 values")
                return False
            if sum(config["metrics"]["bleu_weights"]) != 1.0:
                logger.error("bleu_weights must sum to 1.0")
                return False

            # Validate unit settings
            if not isinstance(config["unit"]["heatmap_size"], list) or len(config["unit"]["heatmap_size"]) != 2:
                logger.error("heatmap_size must be a list of 2 integers")
                return False
            if not isinstance(config["unit"]["target_size"], list) or len(config["unit"]["target_size"]) != 2:
                logger.error("target_size must be a list of 2 integers")
                return False

            # Validate GUI settings
            if not isinstance(config["gui"]["window_size"], list) or len(config["gui"]["window_size"]) != 2:
                logger.error("window_size must be a list of 2 integers")
                return False
            if not isinstance(config["gui"]["start_position"], list) or len(config["gui"]["start_position"]) != 2:
                logger.error("start_position must be a list of 2 integers")
                return False

            logger.info("Configuration validation successful")
            return True
        except KeyError as e:
            logger.error(f"Missing configuration key: {str(e)}")
            return False

    def update_config(self, key: str, value: Any, config: Dict[str, Any] = None) -> None:
        """
        Update a configuration value with validation.

        Args:
            key (str): Configuration key in format 'section.subkey'
            value (Any): New value for the key
            config (Dict[str, Any], optional): Configuration to update. Defaults to self.config.
        """
        if config is None:
            config = self.config

        try:
            section, subkey = key.split(".")
            if section not in config:
                logger.error(f"Invalid section: {section}")
                return

            # Type checking
            expected_type = type(self.DEFAULT_CONFIG[section][subkey])
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                return

            # Range checking for specific keys
            if key == "model.dropout_rate" and not 0 <= value <= 1:
                logger.error("dropout_rate must be between 0 and 1")
                return
            if key in ["dataset.max_length", "dataset.batch_size", "model.num_layers",
                       "model.num_heads", "model.hidden_size", "training.epochs"] and value <= 0:
                logger.error(f"{key} must be positive")
                return

            config[section][subkey] = value
            logger.info(f"Updated {key} to {value}")
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to update {key}: {str(e)}")

    def save_config(self, config: Dict[str, Any], config_path: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config (Dict[str, Any]): Configuration dictionary to save.
            config_path (str): Path to save the configuration.
        """
        try:
            # Resolve path relative to project_root
            config_path = os.path.join(config["general"]["project_root"], config_path)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as file:
                yaml.safe_dump(config, file, default_flow_style=False)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            raise

    def get_dataset_options(self, config: Dict[str, Any] = None) -> List[str]:
        """
        Get available dataset options.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to self.config.

        Returns:
            List[str]: List of available datasets.
        """
        if config is None:
            config = self.config
        try:
            datasets = config["dataset"]["available_datasets"]
            logger.info("Retrieved dataset options")
            return datasets
        except KeyError as e:
            logger.error(f"Failed to get dataset options: {str(e)}")
            return []

    def log_config(self, config: Dict[str, Any] = None) -> None:
        """
        Log the current configuration.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to self.config.
        """
        if config is None:
            config = self.config
        logger.info("Current configuration:")
        for section, settings in config.items():
            logger.info(f"Section: {section}")
            for key, value in settings.items():
                logger.info(f"  {key}: {value}")

    def reset_to_default(self, config: Dict[str, Any] = None) -> None:
        """
        Reset configuration to default values.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to self.config.
        """
        if config is None:
            config = self.config
        config.clear()
        config.update(deepcopy(self.DEFAULT_CONFIG))
        logger.info("Configuration reset to default values")

    def export_config(self, config: Dict[str, Any], path: str) -> None:
        """
        Export configuration to a specified path.

        Args:
            config (Dict[str, Any]): Configuration dictionary to export.
            path (str): Export path relative to project_root.
        """
        try:
            export_path = os.path.join(config["general"]["project_root"], path)
            self.save_config(config, export_path)
            logger.info(f"Configuration exported to {export_path}")
        except Exception as e:
            logger.error(f"Failed to export configuration: {str(e)}")
            raise

    def get_absolute_path(self, relative_path: str) -> str:
        """
        Convert relative path to absolute path using project_root.

        Args:
            relative_path (str): Relative path from config.

        Returns:
            str: Absolute path.
        """
        try:
            return os.path.join(self.config["general"]["project_root"], relative_path)
        except KeyError as e:
            logger.error(f"Failed to resolve path: {str(e)}")
            return relative_path

    def get_config_value(self, key: str, config: Dict[str, Any] = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key (str): Configuration key in format 'section.subkey'
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to self.config.

        Returns:
            Any: Configuration value.
        """
        if config is None:
            config = self.config
        try:
            section, subkey = key.split(".")
            return config[section][subkey]
        except (ValueError, KeyError) as e:
            logger.error(f"Failed to get config value for {key}: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    config_manager = ConfigManager()
    config_manager.log_config()
    config_manager.update_config("training.learning_rate", 0.0002)
    config_manager.save_config(config_manager.config, "config_updated.yaml")
    datasets = config_manager.get_dataset_options()
    logger.info(f"Available datasets: {datasets}")