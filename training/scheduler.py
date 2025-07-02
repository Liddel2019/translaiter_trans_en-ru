import os
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, Any, Optional
from pathlib import Path
from utils.config import ConfigManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class LearningRateScheduler:
    def __init__(self, optimizer: optim.Optimizer, config: Dict[str, Any]):
        self.config_manager = ConfigManager()
        self.config = config
        self.optimizer = optimizer
        self.scheduler = None
        self.scheduler_type = None
        self.step_size = None
        self.gamma = None
        self.min_lr = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.scheduler_type = self.config_manager.get_config_value(
                "scheduler.type", config, default="StepLR"
            )
            self.validate_config_value("scheduler_type", self.scheduler_type, str, non_empty=True)
            if self.scheduler_type not in ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"]:
                raise ValueError(f"Scheduler type {self.scheduler_type} not supported")

            self.step_size = self.config_manager.get_config_value(
                "scheduler.step_size", config, default=10
            )
            self.validate_config_value("step_size", self.step_size, int, positive=True)

            self.gamma = self.config_manager.get_config_value(
                "scheduler.gamma", config, default=0.1
            )
            self.validate_config_value("scheduler.gamma", self.gamma, float, range_bounds=(0, 1))

            self.min_lr = self.config_manager.get_config_value(
                "scheduler.min_lr", config, default=0.00001
            )
            self.validate_config_value("min_lr", self.min_lr, float, positive=True)

            self.patience = self.config_manager.get_config_value(
                "scheduler.patience", config, default=5
            )
            self.validate_config_value("patience", self.patience, int, positive=True)

            self.t_max = self.config_manager.get_config_value(
                "scheduler.t_max", config, default=10
            )
            self.validate_config_value("t_max", self.t_max, int, positive=True)

        except Exception as e:
            logger.error(f"Failed to initialize scheduler configuration: {str(e)}")
            raise ValueError(f"Scheduler configuration initialization failed: {str(e)}")

        self.validate_optimizer()
        self.initialize_scheduler()

    def validate_config_value(self, key: str, value: Any, expected_type: type,
                              non_empty: bool = False, positive: bool = False,
                              range_bounds: Optional[tuple] = None) -> None:
        try:
            if not isinstance(value, expected_type):
                raise ValueError(f"Invalid type for {key}")
            if non_empty and isinstance(value, str) and not value.strip():
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"{key} must be positive")
            if range_bounds and isinstance(value, (int, float)):
                min_val, max_val = range_bounds
                if not (min_val < value <= max_val):
                    raise ValueError(f"{key} out of range")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

    def validate_optimizer(self) -> None:
        try:
            if not isinstance(self.optimizer, optim.Optimizer):
                raise ValueError("Invalid optimizer type")
            if not self.optimizer.param_groups:
                raise ValueError("Optimizer has no parameter groups")
        except Exception as e:
            logger.error(f"Optimizer validation failed: {str(e)}")
            raise ValueError(f"Optimizer validation failed: {str(e)}")

    def initialize_scheduler(self) -> None:
        try:
            if self.scheduler_type == "StepLR":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=self.step_size,
                    gamma=self.gamma
                )
            elif self.scheduler_type == "ReduceLROnPlateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=self.gamma,
                    patience=self.patience,
                    min_lr=self.min_lr
                )
            elif self.scheduler_type == "CosineAnnealingLR":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.t_max,
                    eta_min=self.min_lr
                )
            else:
                raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
        except Exception as e:
            logger.error(f"Failed to initialize scheduler: {str(e)}")
            raise ValueError(f"Scheduler initialization failed: {str(e)}")

    def step(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        try:
            if self.scheduler_type == "ReduceLROnPlateau":
                if metrics is None or 'val_loss' not in metrics:
                    raise ValueError("Missing validation loss for ReduceLROnPlateau")
                self.scheduler.step(metrics['val_loss'])
            else:
                self.scheduler.step()
        except Exception as e:
            logger.error(f"Failed to step scheduler: {str(e)}")
            raise ValueError(f"Scheduler step failed: {str(e)}")

    def get_current_lr(self) -> float:
        try:
            lr = self.optimizer.param_groups[0]['lr']
            return lr
        except Exception as e:
            logger.error(f"Failed to get current learning rate: {str(e)}")
            raise ValueError(f"Failed to get current learning rate: {str(e)}")

    def save_state(self, save_path: str) -> None:
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
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {str(e)}")
            raise ValueError(f"Scheduler state saving failed: {str(e)}")

    def load_state(self, load_path: str) -> None:
        try:
            load_path = self.config_manager.get_absolute_path(load_path)
            checkpoint = torch.load(load_path, map_location=self.device)
            self.scheduler_type = checkpoint['scheduler_type']
            self.step_size = checkpoint['step_size']
            self.gamma = checkpoint['gamma']
            self.min_lr = checkpoint['min_lr']
            self.patience = checkpoint.get('patience', 5)
            self.t_max = checkpoint.get('t_max', 10)
            self.initialize_scheduler()
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {str(e)}")
            raise ValueError(f"Scheduler state loading failed: {str(e)}")

    def validate_scheduler_config(self) -> None:
        try:
            self.validate_config_value("scheduler_type", self.scheduler_type, str, non_empty=True)
            self.validate_config_value("step_size", self.step_size, int, positive=True)
            self.validate_config_value("gamma", self.gamma, float, range_bounds=(0, 1))
            self.validate_config_value("min_lr", self.min_lr, float, positive=True)
            self.validate_config_value("patience", self.patience, int, positive=True)
            self.validate_config_value("t_max", self.t_max, int, positive=True)
            if self.scheduler_type not in ["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"]:
                raise ValueError(f"Scheduler type {self.scheduler_type} not supported")
        except Exception as e:
            logger.error(f"Scheduler configuration validation failed: {str(e)}")
            raise ValueError(f"Scheduler configuration validation failed: {str(e)}")

    def adjust_learning_rate(self, new_lr: float) -> None:
        try:
            if not isinstance(new_lr, float) or new_lr <= 0:
                raise ValueError("Invalid new learning rate")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        except Exception as e:
            logger.error(f"Failed to adjust learning rate: {str(e)}")
            raise ValueError(f"Learning rate adjustment failed: {str(e)}")

    def get_scheduler_state(self) -> Dict[str, Any]:
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
            return state
        except Exception as e:
            logger.error(f"Failed to get scheduler state: {str(e)}")
            raise ValueError(f"Scheduler state retrieval failed: {str(e)}")

    def reset_scheduler(self) -> None:
        try:
            self.initialize_scheduler()
        except Exception as e:
            logger.error(f"Failed to reset scheduler: {str(e)}")
            raise ValueError(f"Scheduler reset failed: {str(e)}")

    def log_scheduler_config(self) -> None:
        try:
            pass
        except Exception as e:
            logger.error(f"Failed to log scheduler configuration: {str(e)}")
            raise ValueError(f"Scheduler configuration logging failed: {str(e)}")

if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        params = [torch.nn.Parameter(torch.randn(10, requires_grad=True))]
        optimizer = torch.optim.AdamW(params, lr=config_manager.get_config_value("training.learning_rate", config,
                                                                                 default=0.0001))
        scheduler = LearningRateScheduler(optimizer, config)
        if scheduler.scheduler_type == "ReduceLROnPlateau":
            scheduler.step(metrics={'val_loss': 0.5})
        else:
            scheduler.step()
        save_path = "scheduler_state_test.pth"
        scheduler.save_state(save_path)
        scheduler.adjust_learning_rate(0.00005)
        scheduler.load_state(save_path)
        state = scheduler.get_scheduler_state()
        scheduler.reset_scheduler()
    except Exception as e:
        logger.error(f"Scheduler test execution failed: {str(e)}")