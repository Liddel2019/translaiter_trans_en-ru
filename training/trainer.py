import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from pathlib import Path
from utils.config import ConfigManager
from utils.logger import Logger
from model.transformer import TransformerModel
from data.dataset import TranslationDataset
import numpy as np
from training.scheduler import LearningRateScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import time

class Trainer:
    def __init__(self, config: Dict[str, Any], model: TransformerModel, dataset: TranslationDataset):
        self.config_manager = ConfigManager()
        self.logger = Logger(config)
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_distributed = False
        self.rank = 0
        self.world_size = 1
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.progress_callback = None

        try:
            self.learning_rate = self.config_manager.get_config_value(
                "training.learning_rate", config, default=0.0001
            )
            self.validate_config_value("learning_rate", self.learning_rate, float, positive=True)

            self.epochs = self.config_manager.get_config_value(
                "training.epochs", config, default=10
            )
            self.validate_config_value("epochs", self.epochs, int, positive=True)

            self.log_steps = self.config_manager.get_config_value(
                "training.log_steps", config, default=100
            )
            self.validate_config_value("log_steps", self.log_steps, int, positive=True)

            self.beam_size = self.config_manager.get_config_value(
                "training.beam_size", config, default=5
            )
            self.validate_config_value("beam_size", self.beam_size, int, positive=True)

            self.sample_size = self.config_manager.get_config_value(
                "training.sample_size", config, default=10
            )
            self.validate_config_value("sample_size", self.sample_size, int, positive=True)

            self.optimizer_type = self.config_manager.get_config_value(
                "training.optimizer_type", config, default="AdamW"
            )
            self.validate_config_value("optimizer_type", self.optimizer_type, str, non_empty=True)
            if self.optimizer_type not in ["AdamW", "Adam", "SGD"]:
                raise ValueError(f"Optimizer type {self.optimizer_type} not supported")

            self.gradient_accumulation_steps = self.config_manager.get_config_value(
                "training.gradient_accumulation_steps", config, default=1
            )
            self.validate_config_value("gradient_accumulation_steps", self.gradient_accumulation_steps, int, positive=True)

            self.checkpoint_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value(
                    "model.checkpoint_path", config, default="model/checkpoints"
                )
            )
            self.validate_config_value("checkpoint_path", self.checkpoint_path, str, non_empty=True)

        except Exception as e:
            self.logger.log_exception(e, "Failed to initialize trainer configuration")
            raise ValueError(f"Trainer configuration initialization failed: {str(e)}")

        self.validate_model()
        self.validate_dataset()
        self.setup_training_environment()

        try:
            self.scheduler = LearningRateScheduler(self.optimizer, self.config)
            self.validate_scheduler_integration()
        except Exception as e:
            self.logger.log_exception(e, "Scheduler initialization failed")
            raise RuntimeError(f"Scheduler initialization failed: {str(e)}")

    def validate_config_value(self, key: str, value: Any, expected_type: type, non_empty: bool = False,
                             positive: bool = False) -> None:
        try:
            if not isinstance(value, expected_type):
                raise ValueError(f"Invalid type for {key}")
            if non_empty and isinstance(value, str) and not value.strip():
                raise ValueError(f"{key} cannot be empty")
            if positive and isinstance(value, (int, float)) and value <= 0:
                raise ValueError(f"{key} must be positive")
        except Exception as e:
            self.logger.log_exception(e, f"Validation failed for {key}")
            raise

    def validate_model(self) -> None:
        try:
            if not isinstance(self.model, TransformerModel):
                raise ValueError("Invalid model type")
            if not hasattr(self.model, 'forward') or not hasattr(self.model, 'generate'):
                raise ValueError("Model not properly initialized")
        except Exception as e:
            self.logger.log_exception(e, "Model validation failed")
            raise ValueError(f"Model validation failed: {str(e)}")

    def validate_dataset(self) -> None:
        try:
            if not isinstance(self.dataset, TranslationDataset):
                raise ValueError("Invalid dataset type")
            if not self.dataset.validate_dataset():
                raise ValueError("Dataset is invalid")
            if self.dataset.get_dataset_size() == 0:
                raise ValueError("Dataset is empty")
        except Exception as e:
            self.logger.log_exception(e, "Dataset validation failed")
            raise ValueError(f"Dataset validation failed: {str(e)}")

    def validate_scheduler_integration(self) -> None:
        try:
            if not self.scheduler:
                raise RuntimeError("Scheduler is not initialized")
            if not isinstance(self.scheduler, LearningRateScheduler):
                raise RuntimeError("Invalid scheduler type")
            self.scheduler.validate_scheduler_config()
        except Exception as e:
            self.logger.log_exception(e, "Scheduler integration validation failed")
            raise RuntimeError(f"Scheduler integration validation failed: {str(e)}")

    def validate_training_params(self) -> None:
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
                raise ValueError(f"Optimizer type {self.optimizer_type} not supported")
        except Exception as e:
            self.logger.log_exception(e, "Training parameters validation failed")
            raise ValueError(f"Training parameters validation failed: {str(e)}")

    def setup_training_environment(self) -> None:
        try:
            self.setup_distributed()
            self.setup_optimizer()
        except Exception as e:
            self.logger.log_exception(e, "Failed to setup training environment")
            raise ValueError(f"Training environment setup failed: {str(e)}")

    def setup_distributed(self) -> None:
        try:
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                dist.init_process_group(backend='nccl')
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
                self.is_distributed = True
                self.model = DDP(self.model.to(self.device), device_ids=[self.rank])
            else:
                self.model = self.model.to(self.device)
        except Exception as e:
            self.logger.log_exception(e, "Failed to setup distributed training")
            raise ValueError(f"Distributed training setup failed: {str(e)}")

    def setup_optimizer(self) -> None:
        try:
            if self.optimizer_type == "AdamW":
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == "Adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif self.optimizer_type == "SGD":
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
            else:
                raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        except Exception as e:
            self.logger.log_exception(e, "Failed to setup optimizer")
            raise ValueError(f"Optimizer setup failed: {str(e)}")

    def optimize_model(self) -> None:
        try:
            if torch.__version__ >= "2.0":
                self.model = torch.compile(self.model)
            else:
                self.logger.log_message("WARNING", "torch.compile not available, skipping model optimization")
        except Exception as e:
            self.logger.log_exception(e, "Failed to optimize model")
            raise ValueError(f"Model optimization failed: {str(e)}")

    def sync_scheduler_state(self) -> None:
        try:
            if self.is_distributed:
                state_dict = self.scheduler.get_scheduler_state()
                state_tensor = torch.tensor([state_dict['current_lr']], device=self.device)
                dist.broadcast(state_tensor, src=0)
                if self.rank != 0:
                    self.scheduler.adjust_learning_rate(state_tensor.item())
        except Exception as e:
            self.logger.log_exception(e, "Failed to synchronize scheduler state")
            raise RuntimeError(f"Scheduler state synchronization failed: {str(e)}")

    def adjust_learning_rate(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.scheduler.step(metrics=metrics)
        except Exception as e:
            self.logger.log_exception(e, "Failed to adjust learning rate")
            raise ValueError(f"Learning rate adjustment failed: {str(e)}")

    def start_training(self, progress_callback: Optional[Callable[[float, Dict[str, Any]], None]] = None) -> None:
        try:
            self.progress_callback = progress_callback
            self.total_epochs = self.config_manager.get_config_value("training.epochs", self.config, default=10)
            self.scheduler.initialize_scheduler()
            self.sync_scheduler_state()

            splits = self.dataset.split_dataset()
            train_data = splits['train']
            val_data = splits['val']

            for epoch in range(self.current_epoch, self.total_epochs):
                self.current_epoch = epoch + 1
                self.model.train()
                total_loss = 0.0
                steps = 0
                accumulated_steps = 0

                self.dataset.processed_data = train_data
                self.dataset.create_batches()

                for batch_idx, batch in enumerate(self.dataset.batches):
                    self.optimizer.zero_grad(set_to_none=True)
                    src_ids = batch['en'].to(self.device)
                    tgt_ids = batch['ru'].to(self.device)
                    self.logger.log_message("INFO", f"src_ids shape: {src_ids.shape}, tgt_ids shape: {tgt_ids.shape}")
                    output = self.model(src_ids, tgt_ids[:, :-1])
                    loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), tgt_ids[:, 1:].view(-1))
                    loss = loss / self.gradient_accumulation_steps

                    loss.backward()

                    accumulated_steps += 1
                    if accumulated_steps >= self.gradient_accumulation_steps:
                        self.optimizer.step()
                        accumulated_steps = 0

                    total_loss += loss.item() * self.gradient_accumulation_steps
                    steps += 1

                    if batch_idx % self.log_steps == 0 and batch_idx > 0:
                        avg_loss = total_loss / steps
                        self.on_training_update({
                            'epoch': epoch + 1,
                            'step': batch_idx,
                            'loss': avg_loss,
                            'learning_rate': self.scheduler.get_current_lr()
                        })

                val_metrics = self.validate(val_data)
                scheduler_type = self.config_manager.get_config_value("scheduler.type", self.config, default="StepLR")
                self.adjust_learning_rate(metrics=val_metrics if scheduler_type == "ReduceLROnPlateau" else None)

                if self.rank == 0:
                    self.save_checkpoint(epoch=epoch + 1)

                progress = (epoch + 1) / self.total_epochs
                self.on_training_update({
                    'epoch': epoch + 1,
                    'progress': progress,
                    'loss': avg_loss,
                    'val_metrics': val_metrics,
                    'learning_rate': self.scheduler.get_current_lr()
                })

        except Exception as e:
            self.logger.log_exception(e, "Training failed")
            raise ValueError(f"Training failed: {str(e)}")

    def validate(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
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
                    output = self.model(src_ids, tgt_ids[:, :-1])
                    loss = nn.CrossEntropyLoss()(output.view(-1, output.size(-1)), tgt_ids[:, 1:].view(-1))
                    total_loss += loss.item()
                    steps += 1

            avg_loss = total_loss / steps if steps > 0 else 0.0
            metrics = {'val_loss': avg_loss}
            return metrics
        except Exception as e:
            self.logger.log_exception(e, "Validation failed")
            raise ValueError(f"Validation failed: {str(e)}")

    def stop_training(self) -> None:
        try:
            self.current_epoch = self.config_manager.get_config_value("training.epochs", self.config, default=10)
            if self.is_distributed:
                dist.destroy_process_group()
            if self.rank == 0 and self.scheduler:
                scheduler_state_path = os.path.join(self.checkpoint_path, f"scheduler_state_epoch_{self.current_epoch}.pth")
                self.scheduler.save_state(scheduler_state_path)
        except Exception as e:
            self.logger.log_exception(e, "Failed to stop training or save scheduler state")
            raise ValueError(f"Failed to stop training or save scheduler state: {str(e)}")

    def get_progress(self) -> float:
        try:
            total_epochs = self.config_manager.get_config_value("training.epochs", self.config, default=10)
            progress = self.current_epoch / total_epochs if total_epochs > 0 else 0.0
            return progress
        except Exception as e:
            self.logger.log_exception(e, "Failed to get training progress")
            return 0.0

    def on_training_update(self, metrics: Dict[str, Any]) -> None:
        try:
            if self.progress_callback:
                progress = self.get_progress()
                self.progress_callback(progress, metrics)
        except Exception as e:
            self.logger.log_exception(e, "Failed to invoke progress callback")
            raise ValueError(f"Progress callback failed: {str(e)}")

    def save_checkpoint(self, epoch: int) -> None:
        try:
            if self.rank == 0:
                checkpoint_file = os.path.join(self.checkpoint_path, f"trainer_checkpoint_epoch_{epoch}.pth")
                os.makedirs(self.checkpoint_path, exist_ok=True)
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.get_scheduler_state() if self.scheduler else None,
                }
                torch.save(checkpoint, checkpoint_file)
        except Exception as e:
            self.logger.log_exception(e, "Failed to save checkpoint")
            raise ValueError(f"Checkpoint saving failed: {str(e)}")

    def load_checkpoint(self, checkpoint_file: str) -> None:
        try:
            checkpoint_path = self.config_manager.get_absolute_path(checkpoint_file)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict'] and self.scheduler:
                self.scheduler.load_state(checkpoint_path)
            self.current_epoch = checkpoint['epoch']
        except Exception as e:
            self.logger.log_exception(e, "Failed to load checkpoint")
            raise ValueError(f"Checkpoint loading failed: {str(e)}")

    def get_training_metrics(self) -> Dict[str, Any]:
        try:
            metrics = {
                'current_epoch': self.current_epoch,
                'total_epochs': self.config_manager.get_config_value("training.epochs", self.config, default=10),
                'learning_rate': self.scheduler.get_current_lr() if self.scheduler else self.learning_rate,
                'beam_size': self.beam_size,
                'sample_size': self.sample_size
            }
            return metrics
        except Exception as e:
            self.logger.log_exception(e, "Failed to get training metrics")
            return {}

if __name__ == "__main__":
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        logger = Logger(config)
        model = TransformerModel(config, src_vocab_size=32000, tgt_vocab_size=32000)
        dataset = TranslationDataset(config, "OPUS Tatoeba")

        def sample_progress_callback(progress: float, metrics: Dict[str, Any]) -> None:
            pass

        trainer = Trainer(config, model, dataset)
        trainer.start_training(progress_callback=sample_progress_callback)
        metrics = trainer.get_training_metrics()
        progress = trainer.get_progress()
        trainer.save_checkpoint(epoch=0)

        if trainer.scheduler and trainer.rank == 0:
            scheduler_state_path = os.path.join(trainer.checkpoint_path, "scheduler_state_test.pth")
            trainer.scheduler.save_state(scheduler_state_path)
    except Exception as e:
        logger.log_exception(e, "Test execution failed")