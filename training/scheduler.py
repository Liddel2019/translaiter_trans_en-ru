import os
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional
from pathlib import Path
from utils.config import ConfigManager

# Configure logging for the scheduler module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class LearningRateScheduler:
    """
    Manages learning rate scheduling for the translaiter_trans_en-ru project.
    Supports multiple scheduler types (StepLR, ReduceLROnPlateau, CosineAnnealingLR)
    with configuration loaded via ConfigManager.
    """

    def __init__(self, optimizer: optim.Optimizer, config: Dict[str, Any]):
        """
        Initialize the LearningRateScheduler with an optimizer and configuration.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to adjust learning rate for.
            config (Dict[str, Any]): Configuration dictionary from config.yaml.

        Raises:
            ValueError: If configuration values or optimizer are invalid.

        Example:
            >>> config_manager = ConfigManager()
            >>> optimizer = torch.optim.AdamW(params, lr=0.0001)
            >>> scheduler = LearningRateScheduler(optimizer, config_manager.config)
        """
        self.config_manager = ConfigManager()
        self.config = config
        self.optimizer = optimizer
        self.scheduler = None
        self.scheduler_type = None
        self.step_size = None
        self.gamma = None
        self.min_lr = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing LearningRateScheduler with device: {self.device}")

        # Fetch and validate scheduler configuration
        try:
            self.scheduler_type = self.config_manager.get_config_value(
                "scheduler.type", config, default="StepLR"
            )
            logger.info(f"Loaded scheduler_type: {self.scheduler_type}")
            self.validate_config_value("scheduler_type", self.scheduler_type, str, non_empty=True)
            if self.scheduler_type not in ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"]:
                logger.error(f"Invalid scheduler_type: {self.scheduler_type}")
                raise ValueError(f"Scheduler type {self.scheduler_type} not supported")

            self.step_size = self.config_manager.get_config_value(
                "scheduler.step_size", config, default=10
            )
            logger.info(f"Loaded step_size: {self.step_size}")
            self.validate_config_value("step_size", self.step_size, int, positive=True)

            self.gamma = self.config_manager.get_config_value(
                "scheduler.gamma", config, default=0.1
            )
            logger.info(f"Loaded gamma: {self.gamma}")
            self.validate_config_value("scheduler.gamma", self.gamma, float, range_bounds=(0, 1))

            self.min_lr = self.config_manager.get_config_value(
                "scheduler.min_lr", config, default=0.00001
            )
            logger.info(f"Loaded min_lr: {self.min_lr}")
            self.validate_config_value("min_lr", self.min_lr, float, positive=True)

            self.patience = self.config_manager.get_config_value(
                "scheduler.patience", config, default=5
            )
            logger.info(f"Loaded patience: {self.patience}")
            self.validate_config_value("patience", self.patience, int, positive=True)

            self.t_max = self.config_manager.get_config_value(
                "scheduler.t_max", config, default=10
            )
            logger.info(f"Loaded t_max: {self.t_max}")
            self.validate_config_value("t_max", self.t_max, int, positive=True)

        except Exception as e:
            logger.error(f"Failed to initialize scheduler configuration: {str(e)}")
            raise ValueError(f"Scheduler configuration initialization failed: {str(e)}")

        # Validate optimizer
        self.validate_optimizer()

        # Initialize scheduler
        self.initialize_scheduler()
        logger.info("LearningRateScheduler initialized successfully")

    def validate_config_value(self, key: str, value: Any, expected_type: type,
                              non_empty: bool = False, positive: bool = False,
                              range_bounds: Optional[tuple] = None) -> None:
        """
        Validate a configuration value.

        Args:
            key (str): Configuration key for logging.
            value (Any): Value to validate.
            expected_type (type): Expected type of the value.
            non_empty (bool): If True, ensures string values are non-empty.
            positive (bool): If True, ensures numeric values are positive.
            range_bounds (tuple, optional): Tuple of (min, max) for range validation.

        Raises:
            ValueError: If the value does not meet validation criteria.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.validate_config_value("gamma", 0.1, float, range_bounds=(0, 1))
        """
        try:
            if not isinstance(value, expected_type):
                logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(value)}")
                raise ValueError(f"Invalid type for {key}")
            if non_empty and isinstance(value, str) and not value.strip():
                logger.error(f"{key} cannot be empty")
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                logger.error(f"{key} must be positive")
                raise ValueError(f"{key} must be positive")
            if range_bounds and isinstance(value, (int, float)):
                min_val, max_val = range_bounds
                if not (min_val < value <= max_val):
                    logger.error(f"{key} must be in range ({min_val}, {max_val}]")
                    raise ValueError(f"{key} out of range")
            logger.debug(f"Validated {key}: {value}")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

    def validate_optimizer(self) -> None:
        """
        Validate the optimizer instance.

        Raises:
            ValueError: If the optimizer is invalid.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.validate_optimizer()
        """
        try:
            if not isinstance(self.optimizer, optim.Optimizer):
                logger.error("Optimizer must be an instance of torch.optim.Optimizer")
                raise ValueError("Invalid optimizer type")
            if not self.optimizer.param_groups:
                logger.error("Optimizer has no parameter groups")
                raise ValueError("Optimizer has no parameter groups")
            logger.info("Optimizer validation successful")
        except Exception as e:
            logger.error(f"Optimizer validation failed: {str(e)}")
            raise ValueError(f"Optimizer validation failed: {str(e)}")

    def initialize_scheduler(self) -> None:
        """
        Initialize the PyTorch scheduler based on configuration.

        Raises:
            ValueError: If scheduler initialization fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.initialize_scheduler()
        """
        try:
            if self.scheduler_type == "StepLR":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=self.step_size,
                    gamma=self.gamma
                )
                logger.info(f"Initialized StepLR with step_size: {self.step_size}, gamma: {self.gamma}")
            elif self.scheduler_type == "ReduceLROnPlateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=self.gamma,
                    patience=self.patience,
                    min_lr=self.min_lr
                )
                logger.info(
                    f"Initialized ReduceLROnPlateau with factor: {self.gamma}, patience: {self.patience}, min_lr: {self.min_lr}")
            elif self.scheduler_type == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.t_max,
                    eta_min=self.min_lr
                )
                logger.info(f"Initialized CosineAnnealingLR with T_max: {self.t_max}, eta_min: {self.min_lr}")
            else:
                logger.error(f"Unsupported scheduler type: {self.scheduler_type}")
                raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {str(e)}")
            raise ValueError(f"Scheduler initialization failed: {str(e)}")

    def step(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the learning rate based on the scheduler type.

        Args:
            metrics (Dict[str, Any], optional): Metrics for ReduceLROnPlateau (e.g., {'val_loss': 0.5}).

        Raises:
            ValueError: If scheduler step fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.step(metrics={'val_loss': 0.5})
        """
        try:
            if self.scheduler_type == "ReduceLROnPlateau":
                if metrics is None or 'val_loss' not in metrics:
                    logger.error("ReduceLROnPlateau requires validation loss in metrics")
                    raise ValueError("Missing validation loss for ReduceLROnPlateau")
                self.scheduler.step(metrics['val_loss'])
            else:
                self.scheduler.step()
            current_lr = self.get_current_lr()
            logger.info(f"Learning rate updated to: {current_lr}")
        except Exception as e:
            logger.error(f"Failed to step scheduler: {str(e)}")
            raise ValueError(f"Scheduler step failed: {str(e)}")

    def get_current_lr(self) -> float:
        """
        Get the current learning rate from the optimizer.

        Returns:
            float: Current learning rate.

        Raises:
            ValueError: If retrieving learning rate fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> lr = scheduler.get_current_lr()
            >>> print(lr)
        """
        try:
            lr = self.optimizer.param_groups[0]['lr']
            logger.debug(f"Current learning rate: {lr}")
            return lr
        except Exception as e:
            logger.error(f"Failed to get current learning rate: {str(e)}")
            raise ValueError(f"Failed to get current learning rate: {str(e)}")

    def save_state(self, save_path: str) -> None:
        """
        Save the scheduler state to a file.

        Args:
            save_path (str): Path to save the scheduler state.

        Raises:
            ValueError: If saving state fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.save_state("scheduler_state.pth")
        """
        try:
            save_path = self.config_manager.get_absolute_path(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'scheduler_state_dict': self.scheduler.state_dict(),
                'scheduler_type': self.scheduler_type,
                'step_size': self.step_size,
                'gamma': self.gamma,
                'min_lr': self.min_lr,
                'patience': self.patience,
                't_max': self.t_max
            }, save_path)
            logger.info(f"Scheduler state saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {str(e)}")
            raise ValueError(f"Scheduler state saving failed: {str(e)}")

    def load_state(self, load_path: str) -> None:
        """
        Load the scheduler state from a file.

        Args:
            load_path (str): Path to the scheduler state file.

        Raises:
            ValueError: If loading state fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.load_state("scheduler_state.pth")
        """
        try:
            load_path = self.config_manager.get_absolute_path(load_path)
            checkpoint = torch.load(load_path, map_location=self.device)
            self.scheduler_type = checkpoint['scheduler_type']
            self.step_size = checkpoint['step_size']
            self.gamma = checkpoint['gamma']
            self.min_lr = checkpoint['min_lr']
            self.patience = checkpoint.get('patience', 5)
            self.t_max = checkpoint.get('t_max', 10)

            # Re-initialize scheduler with loaded parameters
            self.initialize_scheduler()
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"Scheduler state loaded from {load_path}")
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {str(e)}")
            raise ValueError(f"Scheduler state loading failed: {str(e)}")

    def validate_scheduler_config(self) -> None:
        """
        Validate all scheduler configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.validate_scheduler_config()
        """
        try:
            self.validate_config_value("scheduler_type", self.scheduler_type, str, non_empty=True)
            self.validate_config_value("step_size", self.step_size, int, positive=True)
            self.validate_config_value("gamma", self.gamma, float, range_bounds=(0, 1))
            self.validate_config_value("min_lr", self.min_lr, float, positive=True)
            self.validate_config_value("patience", self.patience, int, positive=True)
            self.validate_config_value("t_max", self.t_max, int, positive=True)
            if self.scheduler_type not in ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"]:
                logger.error(f"Invalid scheduler_type: {self.scheduler_type}")
                raise ValueError(f"Scheduler type {self.scheduler_type} not supported")
            logger.info("Scheduler configuration validated successfully")
        except Exception as e:
            logger.error(f"Scheduler configuration validation failed: {str(e)}")
            raise ValueError(f"Scheduler configuration validation failed: {str(e)}")

    def adjust_learning_rate(self, new_lr: float) -> None:
        """
        Manually adjust the learning rate of the optimizer.

        Args:
            new_lr (float): New learning rate to set.

        Raises:
            ValueError: If new learning rate is invalid or adjustment fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.adjust_learning_rate(0.00005)
        """
        try:
            if not isinstance(new_lr, float) or new_lr <= 0:
                logger.error("New learning rate must be a positive float")
                raise ValueError("Invalid new learning rate")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Learning rate manually adjusted to: {new_lr}")
        except Exception as e:
            logger.error(f"Failed to adjust learning rate: {str(e)}")
            raise ValueError(f"Learning rate adjustment failed: {str(e)}")

    def get_scheduler_state(self) -> Dict[str, Any]:
        """
        Get the current state of the scheduler.

        Returns:
            Dict[str, Any]: Dictionary containing scheduler state and parameters.

        Raises:
            ValueError: If retrieving scheduler state fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> state = scheduler.get_scheduler_state()
            >>> print(state)
        """
        try:
            state = {
                'scheduler_type': self.scheduler_type,
                'step_size': self.step_size,
                'gamma': self.gamma,
                'min_lr': self.min_lr,
                'patience': self.patience,
                't_max': self.t_max,
                'current_lr': self.get_current_lr(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }
            logger.debug(f"Retrieved scheduler state: {state}")
            return state
        except Exception as e:
            logger.error(f"Failed to get scheduler state: {str(e)}")
            raise ValueError(f"Scheduler state retrieval failed: {str(e)}")

    def reset_scheduler(self) -> None:
        """
        Reset the scheduler to its initial state.

        Raises:
            ValueError: If scheduler reset fails.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.reset_scheduler()
        """
        try:
            self.initialize_scheduler()
            logger.info("Scheduler reset to initial state")
        except Exception as e:
            logger.error(f"Failed to reset scheduler: {str(e)}")
            raise ValueError(f"Scheduler reset failed: {str(e)}")

    def log_scheduler_config(self) -> None:
        """
        Log the current scheduler configuration.

        Example:
            >>> scheduler = LearningRateScheduler(optimizer, config)
            >>> scheduler.log_scheduler_config()
        """
        try:
            logger.info("Current scheduler configuration:")
            logger.info(f"  Scheduler Type: {self.scheduler_type}")
            logger.info(f"  Step Size: {self.step_size}")
            logger.info(f"  Gamma: {self.gamma}")
            logger.info(f"  Min LR: {self.min_lr}")
            logger.info(f"  Patience: {self.patience}")
            logger.info(f"  T_max: {self.t_max}")
            logger.info(f"  Current LR: {self.get_current_lr()}")
        except Exception as e:
            logger.error(f"Failed to log scheduler configuration: {str(e)}")
            raise ValueError(f"Scheduler configuration logging failed: {str(e)}")


if __name__ == "__main__":
    # Example usage for testing
    try:
        config_manager = ConfigManager()
        config = config_manager.config

        # Create a sample optimizer for testing
        params = [torch.nn.Parameter(torch.randn(10, requires_grad=True))]
        optimizer = torch.optim.AdamW(params, lr=config_manager.get_config_value("training.learning_rate", config,
                                                                                 default=0.0001))

        # Initialize scheduler
        scheduler = LearningRateScheduler(optimizer, config)
        scheduler.log_scheduler_config()

        # Test scheduler step
        if scheduler.scheduler_type == "ReduceLROnPlateau":
            scheduler.step(metrics={'val_loss': 0.5})
        else:
            scheduler.step()

        # Test saving and loading state
        save_path = "scheduler_state_test.pth"
        scheduler.save_state(save_path)
        scheduler.adjust_learning_rate(0.00005)
        scheduler.load_state(save_path)

        # Test getting scheduler state
        state = scheduler.get_scheduler_state()
        logger.info(f"Scheduler state: {state}")

        # Test resetting scheduler
        scheduler.reset_scheduler()
        logger.info("Scheduler test completed successfully")
    except Exception as e:
        logger.error(f"Scheduler test execution failed: {str(e)}")