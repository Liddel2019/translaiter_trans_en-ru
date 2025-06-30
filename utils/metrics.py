import os
import logging
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from utils.config import ConfigManager
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.nn.functional as F
import seaborn as sns
import json

# Configure basic logging for the metrics module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Metrics:
    """Manages evaluation metrics for the translaiter_trans_en-ru project with dynamic configuration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Metrics class with configuration from ConfigManager.

        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None, using ConfigManager.

        Raises:
            ValueError: If metric configuration values are invalid.

        Example:
            >>> config_manager = ConfigManager()
            >>> metrics = Metrics(config_manager.config)
        """
        self.config_manager = ConfigManager()
        self.config = config if config is not None else self.config_manager.config
        self.bleu_weights = None
        self.loss_type = None
        self.thresholds = None
        self.plot_path = None

        try:
            self.bleu_weights = self.config_manager.get_config_value(
                "metrics.bleu_weights", self.config, default=[0.25, 0.25, 0.25, 0.25]
            )
            logger.info(f"Loaded bleu_weights: {self.bleu_weights}")
            self.validate_config_value("bleu_weights", self.bleu_weights, list, non_empty=True)

            self.loss_type = self.config_manager.get_config_value(
                "metrics.loss_type", self.config, default="cross_entropy"
            )
            logger.info(f"Loaded loss_type: {self.loss_type}")
            self.validate_config_value("loss_type", self.loss_type, str, non_empty=True)

            self.thresholds = self.config_manager.get_config_value(
                "metrics.thresholds", self.config, default={"bleu": 0.3}
            )
            logger.info(f"Loaded thresholds: {self.thresholds}")
            self.validate_config_value("thresholds", self.thresholds, dict)

            self.plot_path = self.config_manager.get_config_value(
                "metrics.plot_path", self.config, default="logs/plots"
            )
            logger.info(f"Loaded plot_path: {self.plot_path}")
            self.validate_config_value("plot_path", self.plot_path, str, non_empty=True)

            self.validate_metric_params()
            logger.info("Metrics configuration validated successfully")
        except Exception as e:
            logger.error(f"Failed to initialize metrics configuration: {str(e)}")
            raise ValueError(f"Metrics configuration initialization failed: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type,
                            non_empty: bool = False) -> None:
        """
        Validate a configuration value.

        Args:
            key (str): Configuration key for metrics.
            value (Any): Value to validate.
            expected_type (type): Expected type of the value.
            non_empty (bool): If True, ensures string or list values are non-empty.

        Raises:
            ValueError: If the value does not meet validation criteria.

        Example:
            >>> metrics = Metrics()
            >>> metrics.validate_config_value("bleu_weights", [0.25, 0.25, 0.25, 0.25], list, non_empty=True)
        """
        try:
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                raise ValueError(f"Invalid type for {key}")
            if non_empty and isinstance(value, (str, list)) and not value:
                logger.error(f"{key} cannot be empty")
                raise ValueError(f"{key} cannot be empty")
            logger.debug(f"Validated {key}: {value}")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise ValueError(f"Validation failed for {key}: {str(e)}")

    def validate_metric_params(self) -> None:
        """
        Validate all metric configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.

        Example:
            >>> metrics = Metrics()
            >>> metrics.validate_metric_params()
        """
        try:
            if len(self.bleu_weights) != 4:
                logger.error("bleu_weights must be a list of 4 values")
                raise ValueError("bleu_weights must be a list of 4 values")
            if not all(isinstance(w, (int, float)) and w >= 0 for w in self.bleu_weights):
                logger.error("bleu_weights must contain non-negative numbers")
                raise ValueError("bleu_weights must contain non-negative numbers")
            if abs(sum(self.bleu_weights) - 1.0) > 1e-6:
                logger.error("bleu_weights must sum to 1.0")
                raise ValueError("bleu_weights must sum to 1.0")

            valid_loss_types = ["cross_entropy", "mse"]
            if self.loss_type not in valid_loss_types:
                logger.error(f"Invalid loss_type: {self.loss_type}, must be one of {valid_loss_types}")
                raise ValueError(f"Invalid loss_type: {self.loss_type}")

            if not isinstance(self.thresholds.get("bleu"), (int, float)) or self.thresholds.get("bleu", 0) <= 0:
                logger.error("thresholds.bleu must be a positive number")
                raise ValueError("thresholds.bleu must be a positive number")

            plot_path_absolute = self.config_manager.get_absolute_path(self.plot_path)
            os.makedirs(plot_path_absolute, exist_ok=True)
            logger.debug(f"Ensured plot directory exists: {plot_path_absolute}")

            logger.info("Metric configuration parameters validated successfully")
        except Exception as e:
            logger.error(f"Metric configuration validation failed: {str(e)}")
            raise ValueError(f"Metric configuration validation failed: {str(e)}")

    def calculate_bleu(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        Calculate BLEU score for a given reference and hypothesis using config weights.

        Args:
            reference (List[str]): List of reference translations.
            hypothesis (List[str]): List of hypothesis translations.

        Returns:
            float: BLEU score.

        Raises:
            ValueError: If inputs are invalid or BLEU calculation fails.

        Example:
            >>> metrics = Metrics()
            >>> score = metrics.calculate_bleu(reference=["This is a test"], hypothesis=["This is a test"])
            >>> print(score)
        """
        try:
            if not reference or not hypothesis:
                logger.error("Reference or hypothesis list is empty")
                raise ValueError("Reference or hypothesis list is empty")
            if len(reference) != len(hypothesis):
                logger.error("Reference and hypothesis lists must have the same length")
                raise ValueError("Reference and hypothesis lists must have the same length")

            bleu_weights = self.config_manager.get_config_value(
                "metrics.bleu_weights", self.config, default=[0.25, 0.25, 0.25, 0.25]
            )
            logger.debug(f"Using bleu_weights for calculation: {bleu_weights}")

            bleu_scores = []
            smoothing = SmoothingFunction().method1
            for ref, hyp in zip(reference, hypothesis):
                ref_tokens = ref.split()
                hyp_tokens = hyp.split()
                score = sentence_bleu([ref_tokens], hyp_tokens, weights=bleu_weights, smoothing_function=smoothing)
                bleu_scores.append(score)
            avg_bleu = np.mean(bleu_scores)
            logger.info(f"Calculated BLEU score: {avg_bleu:.4f}")
            return avg_bleu
        except Exception as e:
            logger.error(f"Failed to calculate BLEU score: {str(e)}")
            raise ValueError(f"BLEU score calculation failed: {str(e)}")

    def calculate_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate loss based on configured loss_type from config.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.

        Returns:
            float: Computed loss value.

        Raises:
            ValueError: If loss calculation fails or loss_type is unsupported.

        Example:
            >>> metrics = Metrics()
            >>> predictions = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
            >>> targets = torch.tensor([0, 1])
            >>> loss = metrics.calculate_loss(predictions, targets)
            >>> print(loss)
        """
        try:
            if predictions.shape[0] != targets.shape[0]:
                logger.error("Predictions and targets must have the same batch size")
                raise ValueError("Predictions and targets must have the same batch size")

            loss_type = self.config_manager.get_config_value(
                "metrics.loss_type", self.config, default="cross_entropy"
            )
            logger.debug(f"Using loss_type for calculation: {loss_type}")

            if loss_type == "cross_entropy":
                loss = F.cross_entropy(predictions, targets).item()
            elif loss_type == "mse":
                loss = F.mse_loss(predictions, targets).item()
            else:
                logger.error(f"Unsupported loss_type: {loss_type}")
                raise ValueError(f"Unsupported loss_type: {loss_type}")

            logger.info(f"Calculated {loss_type} loss: {loss:.4f}")
            return loss
        except Exception as e:
            logger.error(f"Failed to calculate loss: {str(e)}")
            raise ValueError(f"Loss calculation failed: {str(e)}")

    def visualize_metrics(self, metrics: Dict[str, List[float]], title: str = "Metrics Plot") -> None:
        """
        Visualize metrics and save plots to configured plot_path.

        Args:
            metrics (Dict[str, List[float]]): Dictionary of metric names and their values over time.
            title (str): Title of the plot.

        Raises:
            ValueError: If visualization fails or plot_path is invalid.

        Example:
            >>> metrics = Metrics()
            >>> metrics_dict = {"bleu": [0.1, 0.2, 0.3], "loss": [1.0, 0.8, 0.6]}
            >>> metrics.visualize_metrics(metrics_dict, "Training Metrics")
        """
        try:
            plot_path = self.config_manager.get_config_value(
                "metrics.plot_path", self.config, default="logs/plots"
            )
            plot_path_absolute = self.config_manager.get_absolute_path(plot_path)
            os.makedirs(plot_path_absolute, exist_ok=True)
            logger.debug(f"Preparing to save plot to: {plot_path_absolute}")

            plt.figure(figsize=(10, 6))
            for metric_name, values in metrics.items():
                plt.plot(values, label=metric_name)
            plt.title(title)
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(plot_path_absolute, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(plot_file)
            plt.close()
            logger.info(f"Visualized metrics to: {plot_file}")
        except Exception as e:
            logger.error(f"Failed to visualize metrics: {str(e)}")
            raise ValueError(f"Metrics visualization failed: {str(e)}")

    def export_metrics(self, metrics: Dict[str, Any], export_path: str) -> None:
        """
        Export metrics to a JSON file using configured paths.

        Args:
            metrics (Dict[str, Any]): Metrics to export.
            export_path (str): Path to export the metrics file.

        Raises:
            ValueError: If exporting metrics fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics_dict = {"bleu": 0.4, "loss": 0.5}
            >>> metrics.export_metrics(metrics_dict, "metrics.json")
        """
        try:
            export_path_absolute = self.config_manager.get_absolute_path(export_path)
            os.makedirs(os.path.dirname(export_path_absolute), exist_ok=True)
            with open(export_path_absolute, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics exported to: {export_path_absolute}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")
            raise ValueError(f"Metrics export failed: {str(e)}")

    def validate_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Validate metrics values against configured thresholds.

        Args:
            metrics (Dict[str, Any]): Metrics to validate.

        Returns:
            bool: True if metrics are valid, False otherwise.

        Raises:
            ValueError: If validation fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics_dict = {"bleu": 0.4, "loss": 0.5}
            >>> is_valid = metrics.validate_metrics(metrics_dict)
            >>> print(is_valid)
        """
        try:
            thresholds = self.config_manager.get_config_value(
                "metrics.thresholds", self.config, default={"bleu": 0.3}
            )
            logger.debug(f"Validating metrics with thresholds: {thresholds}")
            for metric_name, value in metrics.items():
                if not isinstance(value, (int, float)) or value < 0:
                    logger.error(f"Invalid value for {metric_name}: {value}, must be non-negative")
                    return False
                if metric_name in thresholds and value < thresholds[metric_name]:
                    logger.warning(f"{metric_name} value {value} below threshold {thresholds[metric_name]}")
            logger.info("Metrics validated successfully")
            return True
        except Exception as e:
            logger.error(f"Metrics validation failed: {str(e)}")
            return False

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics values with formatted output.

        Args:
            metrics (Dict[str, Any]): Metrics to log.

        Raises:
            ValueError: If logging metrics fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics_dict = {"bleu": 0.4, "loss": 0.5}
            >>> metrics.log_metrics(metrics_dict)
        """
        try:
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Metrics: {metrics_str}")
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            raise ValueError(f"Failed to log metrics: {str(e)}")

    def compare_metrics(self, metrics1: Dict[str, Any], metrics2: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare two sets of metrics and compute differences.

        Args:
            metrics1 (Dict[str, Any]): First set of metrics.
            metrics2 (Dict[str, Any]): Second set of metrics.

        Returns:
            Dict[str, float]: Dictionary of metric differences.

        Raises:
            ValueError: If comparison fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics1 = {"bleu": 0.4, "loss": 0.5}
            >>> metrics2 = {"bleu": 0.5, "loss": 0.4}
            >>> diffs = metrics.compare_metrics(metrics1, metrics2)
            >>> print(diffs)
        """
        try:
            differences = {}
            common_keys = set(metrics1.keys()) & set(metrics2.keys())
            for key in common_keys:
                if isinstance(metrics1[key], (int, float)) and isinstance(metrics2[key], (int, float)):
                    differences[key] = metrics2[key] - metrics1[key]
                    logger.debug(f"Compared {key}: difference = {differences[key]:.4f}")
            logger.info(f"Metrics comparison completed: {differences}")
            return differences
        except Exception as e:
            logger.error(f"Failed to compare metrics: {str(e)}")
            raise ValueError(f"Metrics comparison failed: {str(e)}")

    def aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate multiple sets of metrics by computing averages.

        Args:
            metrics_list (List[Dict[str, Any]]): List of metrics dictionaries.

        Returns:
            Dict[str, float]: Aggregated metrics (averages).

        Raises:
            ValueError: If aggregation fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics_list = [{"bleu": 0.4, "loss": 0.5}, {"bleu": 0.5, "loss": 0.4}]
            >>> agg_metrics = metrics.aggregate_metrics(metrics_list)
            >>> print(agg_metrics)
        """
        try:
            if not metrics_list:
                logger.error("Metrics list is empty")
                raise ValueError("Metrics list is empty")
            aggregated = {}
            keys = metrics_list[0].keys()
            for key in keys:
                values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
                if values:
                    aggregated[key] = np.mean(values)
                    logger.debug(f"Aggregated {key}: {aggregated[key]:.4f}")
            logger.info(f"Aggregated metrics: {aggregated}")
            return aggregated
        except Exception as e:
            logger.error(f"Failed to aggregate metrics: {str(e)}")
            raise ValueError(f"Metrics aggregation failed: {str(e)}")

    def plot_heatmap(self, data: np.ndarray, title: str = "Metric Heatmap") -> None:
        """
        Plot a heatmap for a given data array and save to configured plot_path.

        Args:
            data (np.ndarray): Data array for the heatmap.
            title (str): Title of the heatmap.

        Raises:
            ValueError: If heatmap plotting fails.

        Example:
            >>> metrics = Metrics()
            >>> data = np.random.rand(10, 10)
            >>> metrics.plot_heatmap(data, "Sample Heatmap")
        """
        try:
            plot_path = self.config_manager.get_config_value(
                "metrics.plot_path", self.config, default="logs/plots"
            )
            plot_path_absolute = self.config_manager.get_absolute_path(plot_path)
            os.makedirs(plot_path_absolute, exist_ok=True)
            logger.debug(f"Preparing to save heatmap to: {plot_path_absolute}")

            plt.figure(figsize=(8, 6))
            sns.heatmap(data, annot=True, cmap="viridis")
            plt.title(title)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(plot_path_absolute, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(plot_file)
            plt.close()
            logger.info(f"Heatmap saved to: {plot_file}")
        except Exception as e:
            logger.error(f"Failed to plot heatmap: {str(e)}")
            raise ValueError(f"Heatmap plotting failed: {str(e)}")

    def check_thresholds(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """
        Check if metrics meet configured thresholds.

        Args:
            metrics (Dict[str, Any]): Metrics to check.

        Returns:
            Dict[str, bool]: Dictionary indicating if each metric meets its threshold.

        Raises:
            ValueError: If threshold checking fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics_dict = {"bleu": 0.4}
            >>> result = metrics.check_thresholds(metrics_dict)
            >>> print(result)
        """
        try:
            thresholds = self.config_manager.get_config_value(
                "metrics.thresholds", self.config, default={"bleu": 0.3}
            )
            logger.debug(f"Checking thresholds: {thresholds}")
            results = {}
            for metric_name, value in metrics.items():
                if metric_name in thresholds:
                    results[metric_name] = value >= thresholds[metric_name]
                    logger.debug(f"Checked {metric_name}: {value} >= {thresholds[metric_name]} = {results[metric_name]}")
            logger.info(f"Threshold check results: {results}")
            return results
        except Exception as e:
            logger.error(f"Failed to check thresholds: {str(e)}")
            raise ValueError(f"Threshold checking failed: {str(e)}")

    def log_config(self) -> None:
        """
        Log the current metrics configuration.

        Raises:
            ValueError: If logging configuration fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics.log_config()
        """
        try:
            logger.info("Current metrics configuration:")
            logger.info(f"  BLEU Weights: {self.bleu_weights}")
            logger.info(f"  Loss Type: {self.loss_type}")
            logger.info(f"  Thresholds: {self.thresholds}")
            logger.info(f"  Plot Path: {self.plot_path}")
        except Exception as e:
            logger.error(f"Failed to log configuration: {str(e)}")
            raise ValueError(f"Failed to log configuration: {str(e)}")

    def reset_metrics(self) -> None:
        """
        Reset metrics configuration to defaults from ConfigManager.

        Raises:
            ValueError: If resetting configuration fails.

        Example:
            >>> metrics = Metrics()
            >>> metrics.reset_metrics()
        """
        try:
            self.config_manager.reset_to_default(self.config)
            self.bleu_weights = self.config_manager.get_config_value(
                "metrics.bleu_weights", self.config, default=[0.25, 0.25, 0.25, 0.25]
            )
            logger.info(f"Reset bleu_weights: {self.bleu_weights}")
            self.loss_type = self.config_manager.get_config_value(
                "metrics.loss_type", self.config, default="cross_entropy"
            )
            logger.info(f"Reset loss_type: {self.loss_type}")
            self.thresholds = self.config_manager.get_config_value(
                "metrics.thresholds", self.config, default={"bleu": 0.3}
            )
            logger.info(f"Reset thresholds: {self.thresholds}")
            self.plot_path = self.config_manager.get_config_value(
                "metrics.plot_path", self.config, default="logs/plots"
            )
            logger.info(f"Reset plot_path: {self.plot_path}")
            logger.info("Metrics configuration reset to defaults")
        except Exception as e:
            logger.error(f"Failed to reset metrics configuration: {str(e)}")
            raise ValueError(f"Metrics configuration reset failed: {str(e)}")

    def plot_metrics(self, metrics: Dict[str, List[float]], title: str = "Metrics Plot") -> None:
        """
        Plot multiple metrics in subplots and save to configured plot_path.

        Args:
            metrics (Dict[str, List[float]]): Dictionary of metric names and their values over time.
            title (str): Title of the plot.

        Raises:
            ValueError: If plotting fails or plot_path is invalid.

        Example:
            >>> metrics = Metrics()
            >>> metrics_dict = {"bleu": [0.1, 0.2, 0.3], "loss": [1.0, 0.8, 0.6]}
            >>> metrics.plot_metrics(metrics_dict, "Multi-Metrics Plot")
        """
        try:
            plot_path = self.config_manager.get_config_value(
                "metrics.plot_path", self.config, default="logs/plots"
            )
            plot_path_absolute = self.config_manager.get_absolute_path(plot_path)
            os.makedirs(plot_path_absolute, exist_ok=True)
            logger.debug(f"Preparing to save multi-metrics plot to: {plot_path_absolute}")

            n_metrics = len(metrics)
            fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
            if n_metrics == 1:
                axes = [axes]
            for ax, (metric_name, values) in zip(axes, metrics.items()):
                ax.plot(values, label=metric_name)
                ax.set_title(f"{metric_name} over Epochs")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True)
            plt.suptitle(title)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(plot_path_absolute, f"{title.replace(' ', '_')}_{timestamp}.png")
            plt.savefig(plot_file)
            plt.close()
            logger.info(f"Multi-metrics plot saved to: {plot_file}")
        except Exception as e:
            logger.error(f"Failed to plot multi-metrics: {str(e)}")
            raise ValueError(f"Multi-metrics plotting failed: {str(e)}")

    def validate_imports(self) -> bool:
        """
        Validate all required imports for the Metrics module.

        Returns:
            bool: True if all imports are successful, False otherwise.

        Example:
            >>> metrics = Metrics()
            >>> metrics.validate_imports()
        """
        try:
            import numpy
            import torch
            import matplotlib
            import seaborn
            import nltk
            import json
            logger.info("All required Metrics imports validated successfully")
            return True
        except ImportError as e:
            logger.error(f"Metrics import validation failed: {str(e)}")
            return False

if __name__ == "__main__":
    logger.info("Starting metrics.py test execution")
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        metrics = Metrics(config)

        metrics.log_config()

        reference = ["This is a test sentence", "Another test sentence"]
        hypothesis = ["This is a test sentence", "Another sentence test"]
        bleu_score = metrics.calculate_bleu(reference, hypothesis)
        metrics.log_metrics({"bleu": bleu_score})

        predictions = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        targets = torch.tensor([0, 1])
        loss = metrics.calculate_loss(predictions, targets)
        metrics.log_metrics({"loss": loss})

        metrics_dict = {
            "bleu": [bleu_score, bleu_score + 0.1, bleu_score + 0.2],
            "loss": [loss, loss - 0.1, loss - 0.2]
        }
        metrics.visualize_metrics(metrics_dict, "Test Metrics Plot")
        metrics.plot_metrics(metrics_dict, "Test Multi-Metrics Plot")
        metrics.export_metrics({"bleu": bleu_score, "loss": loss}, "test_metrics.json")
        is_valid = metrics.validate_metrics({"bleu": bleu_score, "loss": loss})
        logger.info(f"Metrics validation: {'Valid' if is_valid else 'Invalid'}")
        threshold_results = metrics.check_thresholds({"bleu": bleu_score})
        logger.info(f"Threshold check results: {threshold_results}")
        metrics2 = {"bleu": bleu_score + 0.1, "loss": loss - 0.1}
        differences = metrics.compare_metrics({"bleu": bleu_score, "loss": loss}, metrics2)
        logger.info(f"Metrics differences: {differences}")
        metrics_list = [
            {"bleu": bleu_score, "loss": loss},
            {"bleu": bleu_score + 0.1, "loss": loss - 0.1}
        ]
        aggregated = metrics.aggregate_metrics(metrics_list)
        logger.info(f"Aggregated metrics: {aggregated}")
        data = np.random.rand(10, 10)
        metrics.plot_heatmap(data, "Test Heatmap")
        metrics.reset_metrics()
        metrics.log_config()
        metrics.validate_imports()
        logger.info("Metrics test completed successfully")
    except Exception as e:
        logger.error(f"Metrics test execution failed: {str(e)}")
        sys.exit(1)