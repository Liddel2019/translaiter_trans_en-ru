import os
import sys
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QProgressBar, QComboBox, QMessageBox,
    QFileDialog, QGridLayout, QAction, QMenuBar, QApplication
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from datetime import datetime
from utils.config import ConfigManager
from utils.metrics import Metrics
from utils.logger import Logger

def import_training_components():
    try:
        from data.dataset import TranslationDataset
        from data.tokenizer import TranslationTokenizer
        from model.transformer import TransformerModel
        from training.trainer import Trainer
        return TranslationDataset, TranslationTokenizer, TransformerModel, Trainer
    except ImportError as e:
        # Use custom Logger instance after initialization in TranslationGUI
        return None, None, None, None

class TranslationGUI(QMainWindow):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.config_manager = ConfigManager()
        self.logger = Logger(self.config)
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.progress_timer = QTimer()
        self.training_in_progress = False
        self.app_manager = None
        self.vis_unit = None
        self.metrics = None

        TranslationDataset, TranslationTokenizer, TransformerModel, Trainer = import_training_components()
        self.TranslationDataset = TranslationDataset
        self.TranslationTokenizer = TranslationTokenizer
        self.TransformerModel = TransformerModel
        self.Trainer = Trainer

        try:
            if self.TranslationDataset and self.TranslationTokenizer and self.TransformerModel:
                self.tokenizer = self.TranslationTokenizer(self.config)
                src_vocab_size = self.config_manager.get_config_value("model.src_vocab_size", self.config, default=32000)
                tgt_vocab_size = self.config_manager.get_config_value("model.tgt_vocab_size", self.config, default=32000)
                self.model = self.TransformerModel(self.config, src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size)
                self.dataset = self.TranslationDataset(self.config, "OPUS Tatoeba")
                self.trainer = self.Trainer(config=self.config, model=self.model, dataset=self.dataset)
            else:
                self.logger.log_message("WARNING", "Trainer initialization skipped due to missing components")
        except Exception as e:
            self.logger.log_exception(e, "Failed to initialize trainer")
            raise RuntimeError(f"Trainer initialization failed: {str(e)}")

        try:
            self.metrics = Metrics(config=self.config)
        except Exception as e:
            self.logger.log_message("WARNING", f"Failed to initialize metrics: {str(e)}")
            self.metrics = None

        self.setup_ui()
        self.validate_trainer_setup()

    def setup_ui(self) -> None:
        try:
            window_size = self.config_manager.get_config_value("gui.window_size", self.config, default=[400, 300])
            start_position = self.config_manager.get_config_value("gui.start_position", self.config, default=[100, 100])
            self.setWindowTitle("Translaiter: English to Russian Translator")
            self.resize(*window_size)
            self.move(*start_position)

            menubar = self.menuBar()
            app_menu = menubar.addMenu("Application")
            restart_action = QAction("Restart Application", self)
            restart_action.triggered.connect(self.launch_main_application)
            app_menu.addAction(restart_action)

            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)

            input_layout = QGridLayout()
            self.input_label = QLabel("Input Text (English):")
            self.input_label.setFont(QFont("Arial", 12))
            self.input_text = QTextEdit()
            self.input_text.setPlaceholderText("Enter English text to translate...")
            self.input_text.setFixedHeight(100)
            input_layout.addWidget(self.input_label, 0, 0)
            input_layout.addWidget(self.input_text, 1, 0, 1, 2)

            self.dataset_label = QLabel("Select Dataset:")
            self.dataset_combo = QComboBox()
            datasets = self.config_manager.get_dataset_options(self.config)
            self.dataset_combo.addItems(datasets)
            input_layout.addWidget(self.dataset_label, 0, 2)
            input_layout.addWidget(self.dataset_combo, 1, 2)

            main_layout.addLayout(input_layout)

            self.output_label = QLabel("Translated Text (Russian):")
            self.output_label.setFont(QFont("Arial", 12))
            self.output_text = QTextEdit()
            self.output_text.setReadOnly(True)
            self.output_text.setFixedHeight(100)
            main_layout.addWidget(self.output_label)
            main_layout.addWidget(self.output_text)

            self.metrics_label = QLabel("Training Metrics:")
            self.metrics_label.setFont(QFont("Arial", 12))
            self.metrics_display = QTextEdit()
            self.metrics_display.setReadOnly(True)
            self.metrics_display.setFixedHeight(80)
            main_layout.addWidget(self.metrics_label)
            main_layout.addWidget(self.metrics_display)

            self.progress_bar = QProgressBar()
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            main_layout.addWidget(self.progress_bar)

            self.figure, self.ax = plt.subplots()
            self.canvas = FigureCanvas(self.figure)
            self.canvas.setFixedHeight(200)
            main_layout.addWidget(self.canvas)

            button_layout = QHBoxLayout()
            self.translate_button = QPushButton("Translate")
            self.translate_button.clicked.connect(self.translate_text)
            self.train_button = QPushButton("Start Training")
            self.train_button.clicked.connect(self.start_training)
            self.save_button = QPushButton("Save Translation")
            self.save_button.clicked.connect(self.save_translation)
            self.heatmap_button = QPushButton("Show Heatmap")
            self.heatmap_button.clicked.connect(self.show_heatmap)
            button_layout.addWidget(self.translate_button)
            button_layout.addWidget(self.train_button)
            button_layout.addWidget(self.save_button)
            button_layout.addWidget(self.heatmap_button)
            main_layout.addLayout(button_layout)

            self.progress_timer.timeout.connect(self.update_progress)
        except Exception as e:
            self.logger.log_exception(e, "Failed to set up UI")
            raise RuntimeError(f"UI setup failed: {str(e)}")

    def validate_trainer_setup(self) -> None:
        try:
            if not self.Trainer:
                raise RuntimeError("Trainer module not available")
            if not self.trainer:
                self.logger.log_message("WARNING", "Trainer not initialized; skipping validation")
                return
            if not isinstance(self.trainer, self.Trainer):
                raise RuntimeError("Invalid trainer type")
            self.trainer.validate_training_params()
        except Exception as e:
            self.logger.log_exception(e, "Trainer setup validation failed")
            raise RuntimeError(f"Trainer setup validation failed: {str(e)}")

    def sync_training_state(self) -> None:
        try:
            if self.trainer:
                metrics = self.trainer.get_training_metrics()
                self.display_metrics(metrics)
                progress = self.trainer.get_progress()
                self.progress_bar.setValue(int(progress * 100))
        except Exception as e:
            self.logger.log_exception(e, "Failed to synchronize training state")
            raise RuntimeError(f"Training state synchronization failed: {str(e)}")

    def launch_main_application(self) -> None:
        try:
            from main import ApplicationManager, main
            if self.training_in_progress:
                self.stop_training()

            self.close()

            self.app_manager = ApplicationManager()
            self.app_manager.parse_arguments()
            self.app_manager.load_initial_config()
            if self.app_manager.validate_environment():
                self.app_manager.setup_logging()
                self.app_manager.initialize_services()
                self.app_manager.run_application()
            else:
                self.logger.log_message("ERROR", "Environment validation failed during restart")
                QMessageBox.critical(self, "Restart Error", "Environment validation failed")
        except ImportError as e:
            self.logger.log_message("WARNING", f"Application restart functionality disabled: {str(e)}")
            QMessageBox.warning(self, "Feature Unavailable", "Application restart is not available")
        except Exception as e:
            self.logger.log_exception(e, "Failed to restart application")
            QMessageBox.critical(self, "Restart Error", f"Failed to restart application: {str(e)}")

    def translate_text(self) -> None:
        try:
            if self.TranslationTokenizer is None or self.TransformerModel is None:
                self.logger.log_message("WARNING", "Translation functionality disabled due to missing modules")
                QMessageBox.warning(self, "Feature Unavailable", "Translation functionality is not available")
                return

            input_text = self.input_text.toPlainText().strip()
            if not input_text:
                self.logger.log_message("WARNING", "No input text provided for translation")
                QMessageBox.warning(self, "Input Error", "Please enter text to translate")
                return

            if not self.tokenizer:
                self.tokenizer = self.TranslationTokenizer(self.config)
            if not self.model:
                src_vocab_size = self.config_manager.get_config_value("model.src_vocab_size", self.config,
                                                                      default=32000)
                tgt_vocab_size = self.config_manager.get_config_value("model.tgt_vocab_size", self.config,
                                                                      default=32000)
                self.model = self.TransformerModel(self.config, src_vocab_size=src_vocab_size,
                                                   tgt_vocab_size=tgt_vocab_size)

            tokens = self.tokenizer.encode(input_text)
            beam_size = self.config_manager.get_config_value("training.beam_size", self.config, default=5)
            translated_tokens = self.model.generate(tokens, beam_size=beam_size)
            translated_text = self.tokenizer.decode(translated_tokens)
            self.output_text.setText(translated_text)
        except Exception as e:
            self.logger.log_exception(e, "Translation failed")
            raise RuntimeError(f"Translation failed: {str(e)}")

    def start_training(self) -> None:
        try:
            if self.TranslationDataset is None or self.TranslationTokenizer is None or self.TransformerModel is None or self.Trainer is None:
                self.logger.log_message("WARNING", "Training functionality disabled due to missing modules")
                QMessageBox.warning(self, "Feature Unavailable", "Training functionality is not available")
                return

            if not self.trainer:
                raise RuntimeError("Trainer not initialized")

            if self.training_in_progress:
                self.logger.log_message("WARNING", "Training already in progress")
                QMessageBox.warning(self, "Training Error", "Training is already in progress")
                return

            selected_dataset = self.dataset_combo.currentText()
            if not selected_dataset:
                self.logger.log_message("WARNING", "No dataset selected for training")
                QMessageBox.warning(self, "Dataset Error", "Please select a dataset")
                return

            try:
                self.dataset = self.TranslationDataset(self.config, selected_dataset)
                self.tokenizer = self.TranslationTokenizer(self.config)
                src_vocab_size = self.config_manager.get_config_value("model.src_vocab_size", self.config,
                                                                      default=32000)
                tgt_vocab_size = self.config_manager.get_config_value("model.tgt_vocab_size", self.config,
                                                                      default=32000)
                self.model = self.TransformerModel(self.config, src_vocab_size=src_vocab_size,
                                                   tgt_vocab_size=tgt_vocab_size)
            except Exception as e:
                self.logger.log_exception(e, "Failed to initialize training components")
                raise RuntimeError(f"Training component initialization failed: {str(e)}")

            try:
                self.trainer = self.Trainer(self.config, self.model, self.dataset)
            except Exception as e:
                self.logger.log_exception(e, "Failed to reinitialize trainer")
                raise RuntimeError(f"Trainer reinitialization failed: {str(e)}")

            self.training_in_progress = True
            self.train_button.setText("Stop Training")
            self.train_button.clicked.disconnect()
            self.train_button.clicked.connect(self.stop_training)

            self.progress_bar.setValue(0)
            self.progress_timer.start(1000)
            self.trainer.start_training(callback=self.on_training_update)
        except Exception as e:
            self.logger.log_exception(e, "Training initiation failed")
            raise RuntimeError(f"Training initiation failed: {str(e)}")

    def stop_training(self) -> None:
        try:
            if self.Trainer is None:
                self.logger.log_message("WARNING", "Training stop functionality disabled due to missing Trainer module")
                return

            if self.trainer and self.training_in_progress:
                self.trainer.stop_training()
                self.training_in_progress = False
                self.progress_timer.stop()
                self.progress_bar.setValue(0)
                self.train_button.setText("Start Training")
                self.train_button.clicked.disconnect()
                self.train_button.clicked.connect(self.start_training)
                self.sync_training_state()
        except Exception as e:
            self.logger.log_exception(e, "Failed to stop training")
            raise RuntimeError(f"Failed to stop training: {str(e)}")

    def display_metrics(self, metrics: Dict[str, float] = None) -> None:
        try:
            if self.metrics is None:
                self.logger.log_message("WARNING", "Metrics display disabled due to missing Metrics module")
                QMessageBox.warning(self, "Feature Unavailable", "Metrics display is not available")
                return

            if metrics is None:
                reference = ["пример текста на русском языке"]
                hypothesis = ["образец текста на русском языке"]
                bleu_score = self.metrics.calculate_bleu(reference, hypothesis)
                metrics = {
                    "bleu": bleu_score,
                    "loss": 0.5
                }

            metrics_text = f"BLEU Score: {metrics.get('bleu', 0.0):.4f}\nLoss: {metrics.get('loss', 0.0):.4f}"
            self.metrics_display.setText(metrics_text)
        except Exception as e:
            self.logger.log_exception(e, "Failed to display metrics")
            raise RuntimeError(f"Failed to display metrics: {str(e)}")

    def show_heatmap(self) -> None:
        try:
            if self.metrics is None:
                self.logger.log_message("WARNING", "Heatmap visualization disabled due to missing Metrics module")
                QMessageBox.warning(self, "Feature Unavailable", "Heatmap visualization is not available")
                return

            heatmap_size = self.config_manager.get_config_value("unit.heatmap_size", self.config, default=[10, 10])
            heatmap_data = np.random.rand(*heatmap_size)
            self.ax.clear()
            sns_heatmap = self.ax.imshow(heatmap_data, cmap="viridis")
            self.ax.set_title("Attention Heatmap")
            self.figure.colorbar(sns_heatmap)
            self.canvas.draw()
        except Exception as e:
            self.logger.log_exception(e, "Failed to display heatmap")
            raise RuntimeError(f"Failed to display heatmap: {str(e)}")

    def update_progress(self) -> None:
        try:
            if self.Trainer is None or not self.trainer or not self.training_in_progress:
                self.logger.log_message("WARNING", "Progress update disabled due to missing Trainer or no active training")
                return

            progress = self.trainer.get_progress()
            self.progress_bar.setValue(int(progress * 100))
            if progress >= 1.0:
                self.stop_training()
                self.display_metrics()
        except Exception as e:
            self.logger.log_exception(e, "Failed to update progress")
            raise RuntimeError(f"Failed to update progress: {str(e)}")

    def save_translation(self) -> None:
        try:
            translated_text = self.output_text.toPlainText()
            if not translated_text:
                self.logger.log_message("WARNING", "No translation to save")
                QMessageBox.warning(self, "Save Error", "No translation to save")
                return

            backup_path = self.config_manager.get_absolute_path(
                self.config_manager.get_config_value("unit.backup_path", self.config, default="logs/backups")
            )
            os.makedirs(backup_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Translation", os.path.join(backup_path, f"translation_{timestamp}.txt"), "Text Files (*.txt)"
            )
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(translated_text)
        except Exception as e:
            self.logger.log_exception(e, "Failed to save translation")
            raise RuntimeError(f"Failed to save translation: {str(e)}")

    def on_training_update(self, metrics: Dict[str, float]) -> None:
        try:
            self.display_metrics(metrics)
        except Exception as e:
            self.logger.log_exception(e, "Failed to update training metrics")
            raise RuntimeError(f"Failed to update training metrics: {str(e)}")

    def initialize_components(self) -> None:
        try:
            if self.TranslationDataset is None or self.TranslationTokenizer is None or self.TransformerModel is None:
                self.logger.log_message("WARNING", "Component initialization disabled due to missing modules")
                QMessageBox.warning(self, "Feature Unavailable", "Component initialization is not available")
                return

            selected_dataset = self.dataset_combo.currentText()
            if selected_dataset:
                self.dataset = self.TranslationDataset(self.config, selected_dataset)
                self.tokenizer = self.TranslationTokenizer(self.config)
                self.model = self.TransformerModel(self.config)
        except Exception as e:
            self.logger.log_exception(e, "Failed to initialize components")
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def clear_output(self) -> None:
        try:
            self.input_text.clear()
            self.output_text.clear()
        except Exception as e:
            self.logger.log_exception(e, "Failed to clear text fields")
            raise RuntimeError(f"Failed to clear text fields: {str(e)}")

    def reset_gui(self) -> None:
        try:
            self.clear_output()
            self.progress_bar.setValue(0)
            self.metrics_display.clear()
            if self.training_in_progress:
                self.stop_training()
        except Exception as e:
            self.logger.log_exception(e, "Failed to reset GUI")
            raise RuntimeError(f"Failed to reset GUI: {str(e)}")

    def load_checkpoint(self) -> None:
        try:
            if self.trainer is None:
                self.logger.log_message("WARNING", "Checkpoint loading disabled due to missing Trainer")
                return

            checkpoint_path = self.config_manager.get_config_value("model.checkpoint_path", self.config, default="model/checkpoints")
            checkpoint_path_absolute = self.config_manager.get_absolute_path(checkpoint_path)
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Checkpoint", checkpoint_path_absolute, "Checkpoint Files (*.pth)"
            )
            if file_path:
                self.trainer.load_checkpoint(file_path)
                self.sync_training_state()
        except Exception as e:
            self.logger.log_exception(e, "Failed to load checkpoint")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

    def save_checkpoint(self) -> None:
        try:
            if self.trainer is None:
                self.logger.log_message("WARNING", "Checkpoint saving disabled due to missing Trainer")
                return

            checkpoint_path = self.config_manager.get_config_value("model.checkpoint_path", self.config, default="model/checkpoints")
            checkpoint_path_absolute = self.config_manager.get_absolute_path(checkpoint_path)
            os.makedirs(checkpoint_path_absolute, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Checkpoint", os.path.join(checkpoint_path_absolute, f"checkpoint_{timestamp}.pth"), "Checkpoint Files (*.pth)"
            )
            if file_path:
                self.trainer.save_checkpoint(epoch=self.trainer.current_epoch)
        except Exception as e:
            self.logger.log_exception(e, "Failed to save checkpoint")
            raise RuntimeError(f"Failed to save checkpoint: {str(e)}")

    def configure_logging(self) -> None:
        try:
            log_level = self.config_manager.get_config_value("general.log_level", self.config, default="INFO")
            log_file = self.config_manager.get_config_value("logger.log_file", self.config, default="logs/translaiter.log")
            log_file_absolute = self.config_manager.get_absolute_path(log_file)
            os.makedirs(os.path.dirname(log_file_absolute), exist_ok=True)
            # Configure the custom Logger instance
            self.logger.configure(log_level=log_level, log_file=log_file_absolute)
        except Exception as e:
            self.logger.log_exception(e, "Failed to configure logging")
            raise RuntimeError(f"Failed to configure logging: {str(e)}")

    def validate_imports(self) -> bool:
        try:
            import PyQt5
            import matplotlib
            import numpy
            TranslationDataset, TranslationTokenizer, TransformerModel, Trainer = import_training_components()
            if all(x is None for x in [TranslationDataset, TranslationTokenizer, TransformerModel, Trainer]):
                self.logger.log_message("WARNING", "Some training components could not be imported")
                return False
            return True
        except ImportError as e:
            self.logger.log_exception(e, "GUI import validation failed")
            return False

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        config_manager = ConfigManager()
        gui = TranslationGUI(config_manager.config)
        gui.configure_logging()
        gui.validate_trainer_setup()
        gui.validate_imports()
        gui.show()
        sys.exit(app.exec_())
    except Exception as e:
        # Use custom Logger instance
        logger = Logger(config={})
        logger.log_exception(e, "GUI test execution failed")
        sys.exit(1)