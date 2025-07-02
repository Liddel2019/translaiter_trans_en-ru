import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import seaborn as sns
from utils.config import ConfigManager

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class VisualizationUnit:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config_manager = ConfigManager()
        self.config = config if config is not None else self.config_manager.config
        self.heatmap_size = None
        self.heatmap_path = None
        self.backup_path = None
        self.target_size = None

        try:
            self.heatmap_size = self.config_manager.get_config_value(
                "unit.heatmap_size", self.config, default=[10, 10]
            )
            logger.info(f"Loaded heatmap_size: {self.heatmap_size}")
            self.validate_config_value("heatmap_size", self.heatmap_size, list, non_empty=True)

            self.heatmap_path = self.config_manager.get_config_value(
                "unit.heatmap_path", self.config, default="logs/heatmaps"
            )
            logger.info(f"Loaded heatmap_path: {self.heatmap_path}")
            self.validate_config_value("heatmap_path", self.heatmap_path, str, non_empty=True)

            self.backup_path = self.config_manager.get_config_value(
                "unit.backup_path", self.config, default="logs/backups"
            )
            logger.info(f"Loaded backup_path: {self.backup_path}")
            self.validate_config_value("backup_path", self.backup_path, str, non_empty=True)

            self.target_size = self.config_manager.get_config_value(
                "unit.target_size", self.config, default=[5, 5]
            )
            logger.info(f"Loaded target_size: {self.target_size}")
            self.validate_config_value("target_size", self.target_size, list, non_empty=True)

            self.validate_visual_params()
            self.setup_visual_environment()
            logger.info("Visualization configuration validated and environment set up successfully")
        except Exception as e:
            logger.error(f"Failed to initialize visualization configuration: {str(e)}")
            raise ValueError(f"Visualization configuration initialization failed: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type,
                              non_empty: bool = False):
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

    def validate_visual_params(self):
        try:
            if len(self.heatmap_size) != 2 or not all(isinstance(s, int) and s > 0 for s in self.heatmap_size):
                logger.error("heatmap_size must be a list of 2 positive integers")
                raise ValueError("heatmap_size must be a list of 2 positive integers")

            if len(self.target_size) != 2 or not all(isinstance(s, int) and s > 0 for s in self.target_size):
                logger.error("target_size must be a list of 2 positive integers")
                raise ValueError("target_size must be a list of 2 positive integers")

            heatmap_path_absolute = self.config_manager.get_absolute_path(self.heatmap_path)
            try:
                os.makedirs(heatmap_path_absolute, exist_ok=True)
                logger.debug(f"Ensured heatmap directory exists: {heatmap_path_absolute}")
            except Exception as e:
                logger.error(f"Invalid heatmap_path: {str(e)}")
                raise ValueError(f"Invalid heatmap_path: {str(e)}")

            backup_path_absolute = self.config_manager.get_absolute_path(self.backup_path)
            try:
                os.makedirs(backup_path_absolute, exist_ok=True)
                logger.debug(f"Ensured backup directory exists: {backup_path_absolute}")
            except Exception as e:
                logger.error(f"Invalid backup_path: {str(e)}")
                raise ValueError(f"Invalid backup_path: {str(e)}")

            logger.info("Visualization parameters validated successfully")
        except Exception as e:
            logger.error(f"Visualization parameters validation failed: {str(e)}")
            raise ValueError(f"Visualization parameters validation failed: {str(e)}")

    def setup_visual_environment(self):
        try:
            heatmap_path_absolute = self.config_manager.get_absolute_path(self.heatmap_path)
            backup_path_absolute = self.config_manager.get_absolute_path(self.backup_path)
            os.makedirs(heatmap_path_absolute, exist_ok=True)
            os.makedirs(backup_path_absolute, exist_ok=True)
            logger.info(f"Visualization environment set up with heatmap_path: {heatmap_path_absolute}, "
                        f"backup_path: {backup_path_absolute}")
        except Exception as e:
            logger.error(f"Failed to set up visualization environment: {str(e)}")
            raise ValueError(f"Visualization environment setup failed: {str(e)}")

    def generate_heatmap(self, data: np.ndarray, title: str = "Heatmap"):
        try:
            heatmap_size = self.config_manager.get_config_value(
                "unit.heatmap_size", self.config, default=[10, 10]
            )
            logger.debug(f"Generating heatmap with size: {heatmap_size}")

            if not isinstance(data, np.ndarray) or data.shape != tuple(heatmap_size):
                logger.error(f"Data must be a numpy array with shape {heatmap_size}, got {data.shape}")
                raise ValueError(f"Data must match heatmap_size {heatmap_size}")

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(data, annot=True, cmap="viridis", ax=ax)
            ax.set_title(title)
            logger.info(f"Generated heatmap with title: {title}")
            return fig
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {str(e)}")
            raise ValueError(f"Heatmap generation failed: {str(e)}")

    def save_heatmap(self, fig: plt.Figure, filename: str):
        try:
            heatmap_path = self.config_manager.get_config_value(
                "unit.heatmap_path", self.config, default="logs/heatmaps"
            )
            heatmap_path_absolute = self.config_manager.get_absolute_path(heatmap_path)
            os.makedirs(heatmap_path_absolute, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = os.path.join(heatmap_path_absolute, f"{timestamp}_{filename}")

            fig.savefig(full_path)
            plt.close(fig)
            logger.info(f"Heatmap saved to: {full_path}")
        except Exception as e:
            logger.error(f"Failed to save heatmap: {str(e)}")
            raise ValueError(f"Heatmap saving failed: {str(e)}")

    def animate_heatmap_sequence(self, data_sequence: List[np.ndarray], title: str = "Heatmap Animation"):
        try:
            heatmap_size = self.config_manager.get_config_value(
                "unit.heatmap_size", self.config, default=[10, 10]
            )
            heatmap_path = self.config_manager.get_config_value(
                "unit.heatmap_path", self.config, default="logs/heatmaps"
            )
            heatmap_path_absolute = self.config_manager.get_absolute_path(heatmap_path)
            os.makedirs(heatmap_path_absolute, exist_ok=True)

            for data in data_sequence:
                if not isinstance(data, np.ndarray) or data.shape != tuple(heatmap_size):
                    logger.error(f"Each data array must have shape {heatmap_size}, got {data.shape}")
                    raise ValueError(f"Data must match heatmap_size {heatmap_size}")

            fig, ax = plt.subplots(figsize=(8, 6))

            def update(frame):
                ax.clear()
                sns.heatmap(data_sequence[frame], annot=True, cmap="viridis", ax=ax)
                ax.set_title(f"{title} - Frame {frame + 1}")
                return ax

            anim = animation.FuncAnimation(fig, update, frames=len(data_sequence), interval=500)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            anim_path = os.path.join(heatmap_path_absolute, f"{title.replace(' ', '_')}_{timestamp}.gif")
            anim.save(anim_path, writer='pillow')
            plt.close(fig)
            logger.info(f"Heatmap animation saved to: {anim_path}")
        except Exception as e:
            logger.error(f"Failed to generate heatmap animation: {str(e)}")
            raise ValueError(f"Heatmap animation failed: {str(e)}")

    def resize_heatmap(self, data: np.ndarray):
        from scipy.ndimage import zoom
        try:
            target_size = self.config_manager.get_config_value(
                "unit.target_size", self.config, default=[5, 5]
            )
            logger.debug(f"Resizing heatmap to target_size: {target_size}")

            if not isinstance(data, np.ndarray):
                logger.error("Input data must be a numpy array")
                raise ValueError("Input data must be a numpy array")
            if len(target_size) != 2 or not all(isinstance(s, int) and s > 0 for s in target_size):
                logger.error("target_size must be a list of 2 positive integers")
                raise ValueError("target_size must be a list of 2 positive integers")

            zoom_factors = [t / s for t, s in zip(target_size, data.shape)]
            resized_data = zoom(data, zoom_factors, order=1)
            logger.info(f"Resized heatmap from {data.shape} to {resized_data.shape}")
            return resized_data
        except Exception as e:
            logger.error(f"Failed to resize heatmap: {str(e)}")
            raise ValueError(f"Heatmap resizing failed: {str(e)}")

    def backup_heatmap(self, data: np.ndarray, filename: str):
        try:
            backup_path = self.config_manager.get_config_value(
                "unit.backup_path", self.config, default="logs/backups"
            )
            backup_path_absolute = self.config_manager.get_absolute_path(backup_path)
            os.makedirs(backup_path_absolute, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = os.path.join(backup_path_absolute, f"{timestamp}_{filename}")

            np.save(full_path, data)
            logger.info(f"Heatmap data backed up to: {full_path}")
        except Exception as e:
            logger.error(f"Failed to backup heatmap data: {str(e)}")
            raise ValueError(f"Heatmap backup failed: {str(e)}")

    def log_config(self):
        try:
            logger.info("Current visualization configuration:")
            logger.info(f"  Heatmap Size: {self.heatmap_size}")
            logger.info(f"  Heatmap Path: {self.heatmap_path}")
            logger.info(f"  Backup Path: {self.backup_path}")
            logger.info(f"  Target Size: {self.target_size}")
        except Exception as e:
            logger.error(f"Failed to log configuration: {str(e)}")
            raise ValueError(f"Failed to log configuration: {str(e)}")

    def reset_config(self):
        try:
            self.config_manager.reset_to_default(self.config)
            self.heatmap_size = self.config_manager.get_config_value(
                "unit.heatmap_size", self.config, default=[10, 10]
            )
            logger.info(f"Reset heatmap_size: {self.heatmap_size}")
            self.heatmap_path = self.config_manager.get_config_value(
                "unit.heatmap_path", self.config, default="logs/heatmaps"
            )
            logger.info(f"Reset heatmap_path: {self.heatmap_path}")
            self.backup_path = self.config_manager.get_config_value(
                "unit.backup_path", self.config, default="logs/backups"
            )
            logger.info(f"Reset backup_path: {self.backup_path}")
            self.target_size = self.config_manager.get_config_value(
                "unit.target_size", self.config, default=[5, 5]
            )
            logger.info(f"Reset target_size: {self.target_size}")
            logger.info("Visualization configuration reset to defaults")
        except Exception as e:
            logger.error(f"Failed to reset visualization configuration: {str(e)}")
            raise ValueError(f"Visualization configuration reset failed: {str(e)}")

    def plot_distribution(self, data: np.ndarray, title: str = "Data Distribution"):
        try:
            if not isinstance(data, np.ndarray):
                logger.error("Input data must be a numpy array")
                raise ValueError("Input data must be a numpy array")

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(data.flatten(), bins=50, density=True)
            ax.set_title(title)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(True)
            logger.info(f"Generated distribution plot with title: {title}")
            return fig
        except Exception as e:
            logger.error(f"Failed to plot distribution: {str(e)}")
            raise ValueError(f"Distribution plotting failed: {str(e)}")

    def save_distribution(self, fig: plt.Figure, filename: str):
        try:
            heatmap_path = self.config_manager.get_config_value(
                "unit.heatmap_path", self.config, default="logs/heatmaps"
            )
            heatmap_path_absolute = self.config_manager.get_absolute_path(heatmap_path)
            os.makedirs(heatmap_path_absolute, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = os.path.join(heatmap_path_absolute, f"{timestamp}_{filename}")

            fig.savefig(full_path)
            plt.close(fig)
            logger.info(f"Distribution plot saved to: {full_path}")
        except Exception as e:
            logger.error(f"Failed to save distribution plot: {str(e)}")
            raise ValueError(f"Distribution plot saving failed: {str(e)}")

    def generate_comparison_heatmap(self, data1: np.ndarray, data2: np.ndarray,
                                    title: str = "Comparison Heatmap"):
        try:
            heatmap_size = self.config_manager.get_config_value(
                "unit.heatmap_size", self.config, default=[10, 10]
            )
            logger.debug(f"Generating comparison heatmap with size: {heatmap_size}")

            if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
                logger.error("Both inputs must be numpy arrays")
                raise ValueError("Both inputs must be numpy arrays")
            if data1.shape != tuple(heatmap_size) or data2.shape != tuple(heatmap_size):
                logger.error(f"Data arrays must have shape {heatmap_size}")
                raise ValueError(f"Data arrays must have shape {heatmap_size}")

            diff_data = data1 - data2
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(diff_data, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title(title)
            logger.info(f"Generated comparison heatmap with title: {title}")
            return fig
        except Exception as e:
            logger.error(f"Failed to generate comparison heatmap: {str(e)}")
            raise ValueError(f"Comparison heatmap generation failed: {str(e)}")

    def validate_data_shape(self, data: np.ndarray):
        try:
            heatmap_size = self.config_manager.get_config_value(
                "unit.heatmap_size", self.config, default=[10, 10]
            )
            if not isinstance(data, np.ndarray) or data.shape != tuple(heatmap_size):
                logger.error(f"Data shape {data.shape} does not match heatmap_size {heatmap_size}")
                return False
            logger.debug(f"Validated data shape: {data.shape}")
            return True
        except Exception as e:
            logger.error(f"Data shape validation failed: {str(e)}")
            return False

    def normalize_data(self, data: np.ndarray):
        try:
            if not isinstance(data, np.ndarray):
                logger.error("Input data must be a numpy array")
                raise ValueError("Input data must be a numpy array")
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                logger.warning("Data has no range, returning zeros")
                return np.zeros_like(data)
            normalized_data = (data - min_val) / (max_val - min_val)
            logger.info(f"Normalized data to range [0, 1]")
            return normalized_data
        except Exception as e:
            logger.error(f"Failed to normalize data: {str(e)}")
            raise ValueError(f"Data normalization failed: {str(e)}")

    def generate_multi_heatmap(self, data_list: List[np.ndarray], titles: List[str],
                               suptitle: str = "Multi Heatmap"):
        try:
            heatmap_size = self.config_manager.get_config_value(
                "unit.heatmap_size", self.config, default=[10, 10]
            )
            if len(data_list) != len(titles):
                logger.error("Number of data arrays must match number of titles")
                raise ValueError("Number of data arrays must match number of titles")

            for data in data_list:
                if not isinstance(data, np.ndarray) or data.shape != tuple(heatmap_size):
                    logger.error(f"Each data array must have shape {heatmap_size}")
                    raise ValueError(f"Each data array must have shape {heatmap_size}")

            n_plots = len(data_list)
            fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
            if n_plots == 1:
                axes = [axes]
            for ax, data, title in zip(axes, data_list, titles):
                sns.heatmap(data, annot=True, cmap="viridis", ax=ax)
                ax.set_title(title)
            plt.suptitle(suptitle)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            logger.info(f"Generated multi-heatmap with suptitle: {suptitle}")
            return fig
        except Exception as e:
            logger.error(f"Failed to generate multi-heatmap: {str(e)}")
            raise ValueError(f"Multi-heatmap generation failed: {str(e)}")

    def save_multi_heatmap(self, fig: plt.Figure, filename: str):
        try:
            heatmap_path = self.config_manager.get_config_value(
                "unit.heatmap_path", self.config, default="logs/heatmaps"
            )
            heatmap_path_absolute = self.config_manager.get_absolute_path(heatmap_path)
            os.makedirs(heatmap_path_absolute, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_path = os.path.join(heatmap_path_absolute, f"{timestamp}_{filename}")

            fig.savefig(full_path)
            plt.close(fig)
            logger.info(f"Multi-heatmap saved to: {full_path}")
        except Exception as e:
            logger.error(f"Failed to save multi-heatmap: {str(e)}")
            raise ValueError(f"Multi-heatmap saving failed: {str(e)}")

if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        vis_unit = VisualizationUnit(config)

        vis_unit.log_config()

        data = np.random.rand(*vis_unit.heatmap_size)
        fig = vis_unit.generate_heatmap(data, "Test Heatmap")
        vis_unit.save_heatmap(fig, "test_heatmap.png")

        flat_data = np.random.rand(100)
        fig = vis_unit.plot_distribution(flat_data, "Test Distribution")
        vis_unit.save_distribution(fig, "test_distribution.png")

        data_sequence = [np.random.rand(*vis_unit.heatmap_size) for _ in range(5)]
        vis_unit.animate_heatmap_sequence(data_sequence, "Test Animation")

        resized_data = vis_unit.resize_heatmap(data)
        logger.info(f"Resized data shape: {resized_data.shape}")

        vis_unit.backup_heatmap(data, "test_backup.npy")

        data2 = np.random.rand(*vis_unit.heatmap_size)
        fig = vis_unit.generate_comparison_heatmap(data, data2, "Test Comparison Heatmap")
        vis_unit.save_heatmap(fig, "test_comparison_heatmap.png")

        is_valid = vis_unit.validate_data_shape(data)
        logger.info(f"Data shape validation: {'Valid' if is_valid else 'Invalid'}")

        data_list = [np.random.rand(*vis_unit.heatmap_size) for _ in range(3)]
        titles = ["Heatmap 1", "Heatmap 2", "Heatmap 3"]
        fig = vis_unit.generate_multi_heatmap(data_list, titles, "Test Multi Heatmap")
        vis_unit.save_multi_heatmap(fig, "test_multi_heatmap.png")

        normalized_data = vis_unit.normalize_data(data)
        logger.info(f"Normalized data range: [{np.min(normalized_data):.2f}, {np.max(normalized_data):.2f}]")

        vis_unit.reset_config()
        vis_unit.log_config()

        logger.info("Visualization unit test completed successfully")
    except Exception as e:
        logger.error(f"Visualization unit test execution failed: {str(e)}")