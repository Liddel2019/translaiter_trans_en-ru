import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from pathlib import Path
from utils.config import ConfigManager
from model.transformer import TransformerModel
from data.dataset import TranslationDataset
import numpy as np
from training.scheduler import LearningRateScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import time

# Configure logging for the trainer module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Trainer:
    """Manages training of the TransformerModel for the translaiter_trans_en-ru project."""

    def __init__(self, config: Dict[str, Any], model: TransformerModel, dataset: TranslationDataset):
        """
        Initialize the Trainer with configuration, model, dataset, and scheduler using ConfigManager.

        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml.
            model (TransformerModel): The transformer model to train.
            dataset (TranslationDataset): The dataset for training.

        Raises:
            ValueError: If configuration values, model, dataset, or scheduler are invalid.
            RuntimeError: If scheduler initialization fails.

        Example:
            >>> config_manager = ConfigManager()
            >>> model = TransformerModel(config_manager.config, 32000, 32000)
            >>> dataset = TranslationDataset(config_manager.config, "OPUS Tatoeba")
            >>> trainer = Trainer(config_manager.config, model, dataset)
        """
        self.config_manager = ConfigManager()
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.scaler = GradScaler() if torch.cuda.is_available() else None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.progress_callback = None

        # Fetch and validate configuration values
        try:
            self.learning_rate = self.config_manager.get_config_value(
                "training.learning_rate", config, default=0.0001
            )
            logger.info(f"Loaded learning_rate: {self.learning_rate}")
            self.validate_config_value("learning_rate", self.learning_rate, float, positive=True)

            self.epochs = self.config_manager.get_config_value(
                "training.epochs", config, default=10
            )
            logger.info(f"Loaded epochs: {self.epochs}")
            self.validate_config_value("epochs", self.epochs, int, positive=True)

            self.log_steps = self.config_manager.get_config_value(
                "training.log_steps", config, default=100
            )
            logger.info(f"Loaded log_steps: {self.log_steps}")
            self.validate_config_value("log_steps", self.log_steps, int, positive=True)

            self.beam_size = self.config_manager.get_config_value(
                "training.beam_size", config, default=5
            )
            logger.info(f"Loaded beam_size: {self.beam_size}")
            self.validate_config_value("beam_size", self.beam_size, int, positive=True)

            self.sample_size = self.config_manager.get_config_value(
                "training.sample_size", config, default=10
            )
            logger.info(f"Loaded sample_size: {self.sample_size}")
            self.validate_config_value("sample_size", self.sample_size, int, positive=True)

            self.optimizer_type = self.config_manager.get_config_value(
                "training.optimizer_type", config, default="AdamW"
            )
            logger.info(f"Loaded optimizer_type: {self.optimizer_type}")
            self.validate_config_value("optimizer_type", self.optimizer_type, str, non_empty=True)
            if self.optimizer_type not in ["AdamW", "Adam", "SGD"]:
                logger.error(f"Invalid optimizer_type: {self.optimizer_type}")
                raise ValueError(f"Optimizer type {self.optimizer_type} not supported")

            self.gradient_accumulation_steps = self.config_manager.get_config_value(
                "training.gradient_accumulation_steps", config, default=1
            )
            logger.info(f"Loaded gradient_accumulation_steps: {self.gradient_accumulation_steps}")
            self.validate_config_value("gradient_accumulation_steps", self.gradient_accumulation_steps, int, positive=True)

            self.checkpoint_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "model.checkpoint_path", config, default="model/checkpoints"
                )
            )
            logger.info(f"Loaded checkpoint_path: {self.checkpoint_path}")
            self.validate_config_value("checkpoint_path", self.checkpoint_path, str, non_empty=True)

        except Exception as e:
            logger.error(f"Failed to initialize trainer configuration: {str(e)}")
            raise ValueError(f"Trainer configuration initialization failed: {str(e)}")

        # Validate model and dataset
        self.validate_model()
        self.validate_dataset()

        # Setup training environment
        self.setup_training_environment()

        # Initialize scheduler
        try:
            self.scheduler = LearningRateScheduler(self.optimizer, self.config)
            logger.info("Scheduler initialized with optimizer")
            self.validate_scheduler_integration()
        except Exception as e:
            logger.error(f"Scheduler initialization failed: {str(e)}")
            raise RuntimeError(f"Scheduler initialization failed: {str(e)}")

        logger.info(f"Trainer initialized with device: {self.device}")

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False,
                             positive: bool = False) -> None:
        """
        Validate a configuration value.

        Args:
            key (str): Configuration key for logging.
            value (Any): Value to validate.
            expected_type (type): Expected type of the value.
            non_empty (bool): If True, ensures string values are non-empty.
            positive (bool): If True, ensures numeric values are positive.

        Raises:
            ValueError: If the value does not meet validation criteria.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.validate_config_value("learning_rate", 0.0001, float, positive=True)
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
            logger.debug(f"Validated {key}: {value}")
        except Exception as e:
            logger.error(f"Validation failed for {key}: {str(e)}")
            raise

    def validate_model(self) -> None:
        """
        Validate the transformer model.

        Raises:
            ValueError: If the model is invalid or not initialized.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.validate_model()
        """
        try:
            if not isinstance(self.model, TransformerModel):
                logger.error("Model must be an instance of TransformerModel")
                raise ValueError("Invalid model type")
            if not hasattr(self.model, 'forward') or not hasattr(self.model, 'generate'):
                logger.error("Model missing required methods (forward or generate)")
                raise ValueError("Model not properly initialized")
            logger.info("Model validation successful")
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise ValueError(f"Model validation failed: {str(e)}")

    def validate_dataset(self) -> None:
        """
        Validate the dataset.

        Raises:
            ValueError: If the dataset is invalid or empty.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.validate_dataset()
        """
        try:
            if not isinstance(self.dataset, TranslationDataset):
                logger.error("Dataset must be an instance of TranslationDataset")
                raise ValueError("Invalid dataset type")
            if not self.dataset.validate_dataset():
                logger.error("Dataset validation failed")
                raise ValueError("Dataset is invalid")
            if self.dataset.get_dataset_size() == 0:
                logger.error("Dataset is empty")
                raise ValueError("Dataset is empty")
            logger.info("Dataset validation successful")
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            raise ValueError(f"Dataset validation failed: {str(e)}")

    def validate_scheduler_integration(self) -> None:
        """
        Validate the integration of the LearningRateScheduler.

        Raises:
            RuntimeError: If scheduler is not properly initialized.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.validate_scheduler_integration()
        """
        try:
            if not self.scheduler:
                logger.error("Scheduler is not initialized")
                raise RuntimeError("Scheduler is not initialized")
            if not isinstance(self.scheduler, LearningRateScheduler):
                logger.error("Scheduler must be an instance of LearningRateScheduler")
                raise RuntimeError("Invalid scheduler type")
            self.scheduler.validate_scheduler_config()
            logger.info("Scheduler integration validated successfully")
        except Exception as e:
            logger.error(f"Scheduler integration validation failed: {str(e)}")
            raise RuntimeError(f"Scheduler integration validation failed: {str(e)}")

    def validate_training_params(self) -> None:
        """
        Validate all training parameters fetched from ConfigManager.

        Raises:
            ValueError: If any training parameter is invalid.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.validate_training_params()
        """
        try:
            self.validate_config_value("learning_rate", self.learning_rate, float, positive=True)
            self.validate_config_value("epochs", self.epochs, int, positive=True)
            self.validate_config_value("log_steps", self.log_steps, int, positive=True)
            self.validate_config_value("beam_size", self.beam_size, int, positive=True)
            self.validate_config_value("sample_size", self.sample_size, int, positive=True)
            self.validate_config_value("optimizer_type", self.optimizer_type, str, non_empty=True)
            self.validate_config_value("gradient_accumulation_steps", self.gradient_accumulation_steps, int, positive=True)
            self.validate_config_value("checkpoint_path", self.checkpoint_path, str, non_empty=True)
            if self.optimizer_type not in ["AdamW", "Adam", "SGD"]:
                logger.error(f"Invalid optimizer_type: {self.optimizer_type}")
                raise ValueError(f"Optimizer type {self.optimizer_type} not supported")
            logger.info("Training parameters validated successfully")
        except Exception as e:
            logger.error(f"Training parameters validation failed: {str(e)}")
            raise ValueError(f"Training parameters validation failed: {str(e)}")

    def setup_training_environment(self) -> None:
        """
        Setup distributed training, optimizer, and scheduler.

        Raises:
            ValueError: If setup fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.setup_training_environment()
        """
        try:
            self.setup_distributed()
            self.setup_optimizer()
            logger.info("Training environment setup completed")
        except Exception as e:
            logger.error(f"Failed to setup training environment: {str(e)}")
            raise ValueError(f"Training environment setup failed: {str(e)}")

    def setup_distributed(self) -> None:
        """
        Setup distributed training if multiple GPUs are available.

        Raises:
            ValueError: If distributed setup fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.setup_distributed()
        """
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                dist.init_process_group(backend='nccl')
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.is_distributed = True
                self.model = DDP(self.model.to(self.device), device_ids=[self.rank])
                logger.info(f"Initialized distributed training with {self.world_size} GPUs, rank: {self.rank}")
            else:
                self.model = self.model.to(self.device)
                logger.info("Single GPU or CPU training")
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {str(e)}")
            raise ValueError(f"Distributed training setup failed: {str(e)}")

    def setup_optimizer(self) -> None:
        """
        Setup the optimizer based on configuration.

        Raises:
            ValueError: If optimizer setup fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.setup_optimizer()
        """
        try:
            if self.optimizer_type == "AdamW":
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            else:
                logger.error(f"Unsupported optimizer type: {self.optimizer_type}")
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
            logger.info(f"Initialized {self.optimizer_type} optimizer with learning rate: {self.learning_rate}")
        except Exception as e:
            logger.error(f"Failed to setup optimizer: {str(e)}")
            raise ValueError(f"Optimizer setup failed: {str(e)}")

    def optimize_model(self) -> None:
        """
        Optimize the model using torch.compile if supported (PyTorch >= 2.0).

        Raises:
            ValueError: If model optimization fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.optimize_model()
        """
        try:
            if torch.__version__ >= "2.0":
                self.model = torch.compile(self.model)
                logger.info("Model optimized with torch.compile")
            else:
                logger.warning("torch.compile not available, skipping model optimization")
        except Exception as e:
            logger.error(f"Failed to optimize model: {str(e)}")
            raise ValueError(f"Model optimization failed: {str(e)}")

    def sync_scheduler_state(self) -> None:
        """
        Synchronize scheduler state across distributed processes if applicable.

        Raises:
            RuntimeError: If scheduler state synchronization fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.sync_scheduler_state()
        """
        try:
            if self.is_distributed:
                state_dict = self.scheduler.get_scheduler_state()
                state_tensor = torch.tensor([state_dict['current_lr']], device=self.device)
                dist.broadcast(state_tensor, src=0)
                if self.rank != 0:
                    self.scheduler.adjust_learning_rate(state_tensor.item())
                logger.info(f"Scheduler state synchronized across {self.world_size} processes")
        except Exception as e:
            logger.error(f"Failed to synchronize scheduler state: {str(e)}")
            raise RuntimeError(f"Scheduler state synchronization failed: {str(e)}")

    def adjust_learning_rate(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Adjust the learning rate using the scheduler.

        Args:
            metrics (Dict[str, Any], optional): Metrics for scheduler step (e.g., {'val_loss': 0.5}).

        Raises:
            ValueError: If learning rate adjustment fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.adjust_learning_rate(metrics={'val_loss': 0.5})
        """
        try:
            self.scheduler.step(metrics=metrics)
            current_lr = self.scheduler.get_current_lr()
            logger.info(f"Adjusted learning rate to {current_lr}")
        except Exception as e:
            logger.error(f"Failed to adjust learning rate: {str(e)}")
            raise ValueError(f"Learning rate adjustment failed: {str(e)}")

    def start_training(self, progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None) -> None:
        """
        Start the training loop with AMP, distributed training, and scheduler integration.

        Args:
            progress_callback (Callable[[float, Dict[str, Any]], None], optional): Callback for progress updates.

        Raises:
            ValueError: If training fails to start or encounters errors.

        Example:
            >>> config_manager = ConfigManager()
            >>> model = TransformerModel(config_manager.config, 32000, 32000)
            >>> dataset = TranslationDataset(config_manager.config, "OPUS Tatoeba")
            >>> trainer = Trainer(config_manager.config, model, dataset)
            >>> trainer.start_training(progress_callback=lambda p, m: print(f"Progress: {p}, Metrics: {m}"))
        """
        try:
            self.progress_callback = progress_callback
            self.total_epochs = self.config_manager.get_config_value("training.epochs", self.config, default=10)
            logger.info(f"Training setup completed with epochs: {self.total_epochs}, optimizer: {self.optimizer_type}")
            self.scheduler.initialize_scheduler()
            logger.info("Scheduler initialized with optimizer")
            self.sync_scheduler_state()

            # Split dataset
            splits = self.dataset.split_dataset()
            train_data = splits['train']
            val_data = splits['val']
            logger.info(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples")

            for epoch in range(self.current_epoch, self.total_epochs):
                self.current_epoch = epoch + 1
                self.model.train()
                total_loss = 0.0
                steps = 0
                accumulated_steps = 0

                # Create batches for training
                self.dataset.processed_data = train_data
                self.dataset.create_batches()

                for batch_idx, batch in enumerate(self.dataset.batches):
                    self.optimizer.zero_grad(set_to_none=True)
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        src_ids = batch['en'].to(self.device)
                        tgt_ids = batch['ru'].to(self.device)
                        output = self.model(src_ids, tgt_ids[:, :-1])
                        loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), tgt_ids[:, 1:].view(-1))
                        loss = loss / self.gradient_accumulation_steps

                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    accumulated_steps += 1
                    if accumulated_steps >= self.gradient_accumulation_steps:
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        accumulated_steps = 0

                    total_loss += loss.item() * self.gradient_accumulation_steps
                    steps += 1

                    if batch_idx % self.log_steps == 0 and batch_idx > 0:
                        avg_loss = total_loss / steps
                        logger.info(f"Epoch {epoch + 1}/{self.total_epochs}, Step {batch_idx}, Loss: {avg_loss:.4f}")
                        self.on_training_update({
                            'epoch': epoch + 1,
                            'step': batch_idx,
                            'loss': avg_loss,
                            'learning_rate': self.scheduler.get_current_lr()
                        })

                # Validate after each epoch
                val_metrics = self.validate(val_data)
                logger.info(f"Validation Metrics: {val_metrics}")

                # Adjust learning rate based on scheduler type
                scheduler_type = self.config_manager.get_config_value("scheduler.type", self.config, default="StepLR")
                self.adjust_learning_rate(metrics=val_metrics if scheduler_type == "ReduceLROnPlateau" else None)

                # Save checkpoint
                if self.rank == 0:
                    self.save_checkpoint(epoch=epoch + 1)

                # Update progress
                progress = (epoch + 1) / self.total_epochs
                self.on_training_update({
                    'epoch': epoch + 1,
                    'progress': progress,
                    'loss': avg_loss,
                    'val_metrics': val_metrics,
                    'learning_rate': self.scheduler.get_current_lr()
                })

            logger.info("Training completed")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ValueError(f"Training failed: {str(e)}")

    def validate(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Args:
            val_data (List[Dict[str, Any]]): Validation data.

        Returns:
            Dict[str, float]: Validation metrics (e.g., loss).

        Raises:
            ValueError: If validation fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> metrics = trainer.validate(dataset.split_dataset()['val'])
            >>> print(metrics)
        """
        try:
            self.model.eval()
            total_loss = 0.0
            steps = 0
            self.dataset.processed_data = val_data
            self.dataset.create_batches()

            with torch.no_grad():
                for batch in self.dataset.batches:
                    src_ids = batch['en'].to(self.device)
                    tgt_ids = batch['ru'].to(self.device)
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        output = self.model(src_ids, tgt_ids[:, :-1])
                        loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), tgt_ids[:, 1:].view(-1))
                    total_loss += loss.item()
                    steps += 1

            avg_loss = total_loss / steps if steps > 0 else 0.0
            metrics = {'val_loss': avg_loss}
            logger.info(f"Validation completed, average loss: {avg_loss:.4f}")
            return metrics
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise ValueError(f"Validation failed: {str(e)}")

    def stop_training(self) -> None:
        """
        Gracefully stop the training process and save scheduler state.

        Raises:
            ValueError: If stopping training or saving scheduler state fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.stop_training()
        """
        try:
            self.current_epoch = self.config_manager.get_config_value("training.epochs", self.config, default=10)
            if self.is_distributed:
                dist.destroy_process_group()
            if self.rank == 0 and self.scheduler:
                scheduler_state_path = os.path.join(self.checkpoint_path, f"scheduler_state_epoch_{self.current_epoch}.pth")
                self.scheduler.save_state(scheduler_state_path)
                logger.info(f"Scheduler state saved to {scheduler_state_path}")
            logger.info("Training stopped gracefully")
        except Exception as e:
            logger.error(f"Failed to stop training or save scheduler state: {str(e)}")
            raise ValueError(f"Failed to stop training or save scheduler state: {str(e)}")

    def get_progress(self) -> float:
        """
        Get the current training progress (0.0 to 1.0).

        Returns:
            float: Training progress as a fraction.

        Raises:
            ValueError: If retrieving progress fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> progress = trainer.get_progress()
            >>> print(progress)
        """
        try:
            total_epochs = self.config_manager.get_config_value("training.epochs", self.config, default=10)
            progress = self.current_epoch / total_epochs if total_epochs > 0 else 0.0
            logger.debug(f"Current training progress: {progress:.2f}")
            return progress
        except Exception as e:
            logger.error(f"Failed to get training progress: {str(e)}")
            return 0.0

    def on_training_update(self, metrics: Dict[str, Any]) -> None:
        """
        Handle training updates and invoke the progress callback.

        Args:
            metrics (Dict[str, Any]): Metrics to pass to the callback.

        Raises:
            ValueError: If callback invocation fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.on_training_update({'epoch': 1, 'loss': 0.5})
        """
        try:
            if self.progress_callback:
                progress = self.get_progress()
                self.progress_callback(progress, metrics)
                logger.debug(f"Invoked progress callback with metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to invoke progress callback: {str(e)}")
            raise ValueError(f"Progress callback failed: {str(e)}")

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save the current model, optimizer, and scheduler state.

        Args:
            epoch (int): Current epoch number.

        Raises:
            ValueError: If checkpoint saving fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.save_checkpoint(epoch=1)
        """
        try:
            if self.rank == 0:
                checkpoint_file = os.path.join(self.checkpoint_path, f"trainer_checkpoint_epoch_{epoch}.pth")
                os.makedirs(self.checkpoint_path, exist_ok=True)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.get_scheduler_state() if self.scheduler else None,
                    'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
                }
                torch.save(checkpoint, checkpoint_file)
                logger.info(f"Checkpoint saved to {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise ValueError(f"Checkpoint saving failed: {str(e)}")

    def load_checkpoint(self, checkpoint_file: str) -> None:
        """
        Load model, optimizer, and scheduler state from a checkpoint.

        Args:
            checkpoint_file (str): Path to the checkpoint file.

        Raises:
            ValueError: If checkpoint loading fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> trainer.load_checkpoint("model/checkpoints/trainer_checkpoint_epoch_1.pth")
        """
        try:
            checkpoint_path = self.config_manager.get_absolute_path(checkpoint_file)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state(checkpoint_path)
            if checkpoint['scaler_state_dict'] and self.scaler:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            logger.info(f"Loaded checkpoint from {checkpoint_path}, epoch {self.current_epoch}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise ValueError(f"Checkpoint loading failed: {str(e)}")

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Get current training metrics (e.g., current epoch, loss, learning rate).

        Returns:
            Dict[str, Any]: Current training metrics.

        Raises:
            ValueError: If retrieving metrics fails.

        Example:
            >>> trainer = Trainer(config, model, dataset)
            >>> metrics = trainer.get_training_metrics()
            >>> print(metrics)
        """
        try:
            metrics = {
                'current_epoch': self.current_epoch,
                'total_epochs': self.config_manager.get_config_value("training.epochs", self.config, default=10),
                'learning_rate': self.scheduler.get_current_lr() if self.scheduler else self.learning_rate,
                'beam_size': self.beam_size,
                'sample_size': self.sample_size
            }
            logger.debug(f"Retrieved training metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to get training metrics: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage for testing
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        model = TransformerModel(config, src_vocab_size=32000, tgt_vocab_size=32000)
        dataset = TranslationDataset(config, "OPUS Tatoeba")

        def sample_progress_callback(progress: float, metrics: Dict[str, Any]) -> None:
            logger.info(f"Progress: {progress:.2f}, Metrics: {metrics}")

        trainer = Trainer(config, model, dataset)
        logger.info("Trainer configuration:")
        logger.info(f"Learning rate: {trainer.learning_rate}")
        logger.info(f"Epochs: {trainer.epochs}")
        logger.info(f"Log steps: {trainer.log_steps}")
        logger.info(f"Beam size: {trainer.beam_size}")
        logger.info(f"Sample size: {trainer.sample_size}")
        logger.info(f"Optimizer type: {trainer.optimizer_type}")
        logger.info(f"Gradient accumulation steps: {trainer.gradient_accumulation_steps}")
        logger.info(f"Checkpoint path: {trainer.checkpoint_path}")
        logger.info(f"Device: {trainer.device}")
        logger.info(f"Distributed training: {trainer.is_distributed}")
        if trainer.scheduler:
            trainer.scheduler.log_scheduler_config()
            logger.info("Scheduler configuration logged")

        # Test training
        trainer.start_training(progress_callback=sample_progress_callback)

        # Test metrics and progress
        metrics = trainer.get_training_metrics()
        logger.info(f"Training metrics: {metrics}")
        progress = trainer.get_progress()
        logger.info(f"Training progress: {progress:.2f}")

        # Test checkpoint saving
        trainer.save_checkpoint(epoch=0)

        # Test scheduler state saving
        if trainer.scheduler and trainer.rank == 0:
            scheduler_state_path = os.path.join(trainer.checkpoint_path, "scheduler_state_test.pth")
            trainer.scheduler.save_state(scheduler_state_path)
            logger.info(f"Scheduler state saved for testing: {scheduler_state_path}")
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")