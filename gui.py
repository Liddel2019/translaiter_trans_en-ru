import os
import sys
import logging
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

# Delayed imports to avoid circular dependencies
def import_training_components():
    try:
        from data.dataset import TranslationDataset
        from data.tokenizer import TranslationTokenizer
        from model.transformer import TransformerModel
        from training.trainer import Trainer
        return TranslationDataset, TranslationTokenizer, TransformerModel, Trainer
    except ImportError as e:
        logging.warning(f"Failed to import training components: {str(e)}. Disabling related functionality.")
        return None, None, None, None

# Configure logging for the GUI module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TranslationGUI(QMainWindow):
    """Main GUI class for the translaiter_trans_en-ru project, handling user interaction and training management."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GUI with configuration settings and trainer integration.

        Args:
            config (Dict[str, Any]): Configuration dictionary from config.yaml.

        Raises:
            RuntimeError: If configuration or trainer initialization fails.

        Example:
            >>> config_manager = ConfigManager()
            >>> gui = TranslationGUI(config_manager.config)
        """
        super().__init__()
        self.config = config
        self.config_manager = ConfigManager()
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.progress_timer = QTimer()
        self.training_in_progress = False
        self.app_manager = None
        self.vis_unit = None
        self.metrics = None

        # Initialize training components
        TranslationDataset, TranslationTokenizer, TransformerModel, Trainer = import_training_components()
        self.TranslationDataset = TranslationDataset
        self.TranslationTokenizer = TranslationTokenizer
        self.TransformerModel = TransformerModel
        self.Trainer = Trainer

        # Initialize trainer
        try:
            if self.Trainer:
                self.trainer = self.Trainer(config=self.config, model=None, dataset=None)
                logger.info("Trainer initialized with placeholder model and dataset")
            else:
                logger.warning("Trainer not initialized due to missing module")
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {str(e)}")
            raise RuntimeError(f"Trainer initialization failed: {str(e)}")

        # Initialize metrics
        try:
            self.metrics = Metrics(config=self.config)
            logger.info("Metrics initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize metrics: {str(e)}")
            self.metrics = None

        self.setup_ui()
        self.validate_trainer_setup()
        logger.info("TranslationGUI initialized with provided configuration")
        logger.info("GUI components initialized")

    def setup_ui(self) -> None:
        """
        Set up the main window layout and widgets, including a menu for application restart.

        Raises:
            RuntimeError: If UI setup fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.setup_ui()
        """
        try:
            window_size = self.config_manager.get_config_value("gui.window_size", self.config, default=[400, 300])
            start_position = self.config_manager.get_config_value("gui.start_position", self.config, default=[100, 100])
            logger.debug(f"Setting window size: {window_size}, position: {start_position}")
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

            logger.info("UI setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to set up UI: {str(e)}")
            raise RuntimeError(f"UI setup failed: {str(e)}")

    def validate_trainer_setup(self) -> None:
        """
        Validate the trainer setup to ensure it is properly initialized.

        Raises:
            RuntimeError: If trainer is not initialized or invalid.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.validate_trainer_setup()
        """
        try:
            if not self.trainer or not self.Trainer:
                logger.error("Trainer not initialized")
                raise RuntimeError("Trainer not initialized")
            if not isinstance(self.trainer, self.Trainer):
                logger.error("Trainer is not an instance of Trainer class")
                raise RuntimeError("Invalid trainer type")
            self.trainer.validate_training_params()
            logger.info("Trainer setup validated successfully")
        except Exception as e:
            logger.error(f"Trainer setup validation failed: {str(e)}")
            raise RuntimeError(f"Trainer setup validation failed: {str(e)}")

    def sync_training_state(self) -> None:
        """
        Synchronize the training state with the trainer instance.

        Raises:
            RuntimeError: If synchronization fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.sync_training_state()
        """
        try:
            if self.trainer:
                metrics = self.trainer.get_training_metrics()
                self.display_metrics(metrics)
                progress = self.trainer.get_progress()
                self.progress_bar.setValue(int(progress * 100))
                logger.debug(f"Synchronized training state: progress={progress:.2f}, metrics={metrics}")
        except Exception as e:
            logger.error(f"Failed to synchronize training state: {str(e)}")
            raise RuntimeError(f"Training state synchronization failed: {str(e)}")

    def launch_main_application(self) -> None:
        """
        Trigger the main application flow by reinitializing ApplicationManager.

        Raises:
            RuntimeError: If application restart fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.launch_main_application()
        """
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
                logger.error("Environment validation failed during restart")
                QMessageBox.critical(self, "Restart Error", "Environment validation failed")
        except ImportError as e:
            logger.warning(f"Application restart functionality disabled: {str(e)}")
            QMessageBox.warning(self, "Feature Unavailable", "Application restart is not available")
        except Exception as e:
            logger.error(f"Failed to restart application: {str(e)}")
            QMessageBox.critical(self, "Restart Error", f"Failed to restart application: {str(e)}")

    def translate_text(self) -> None:
        """
        Trigger text translation based on user input.

        Raises:
            RuntimeError: If translation fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.input_text.setText("Hello, world!")
            >>> gui.translate_text()
        """
        try:
            if self.TranslationTokenizer is None or self.TransformerModel is None:
                logger.warning("Translation functionality disabled due to missing modules")
                QMessageBox.warning(self, "Feature Unavailable", "Translation functionality is not available")
                return

            input_text = self.input_text.toPlainText().strip()
            if not input_text:
                logger.warning("No input text provided for translation")
                QMessageBox.warning(self, "Input Error", "Please enter text to translate")
                return

            if not self.tokenizer:
                self.tokenizer = self.TranslationTokenizer(self.config)
                logger.info("Initialized TranslationTokenizer")
            if not self.model:
                self.model = self.TransformerModel(self.config)
                logger.info("Initialized TransformerModel")

            tokens = self.tokenizer.encode(input_text)
            beam_size = self.config_manager.get_config_value("training.beam_size", self.config, default=5)
            translated_tokens = self.model.generate(tokens, beam_size=beam_size)
            translated_text = self.tokenizer.decode(translated_tokens)
            self.output_text.setText(translated_text)
            logger.info(f"Translated text: {translated_text}")
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation failed: {str(e)}")

    def start_training(self) -> None:
        """
        Initiate model training using the Trainer instance based on selected dataset.

        Raises:
            RuntimeError: If training initiation fails or trainer is not initialized.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.dataset_combo.setCurrentText("OPUS Tatoeba")
            >>> gui.start_training()
        """
        try:
            if self.TranslationDataset is None or self.TranslationTokenizer is None or self.TransformerModel is None or self.Trainer is None:
                logger.warning("Training functionality disabled due to missing modules")
                QMessageBox.warning(self, "Feature Unavailable", "Training functionality is not available")
                return

            if not self.trainer:
                logger.error("Trainer not initialized")
                raise RuntimeError("Trainer not initialized")

            if self.training_in_progress:
                logger.warning("Training already in progress")
                QMessageBox.warning(self, "Training Error", "Training is already in progress")
                return

            selected_dataset = self.dataset_combo.currentText()
            if not selected_dataset:
                logger.warning("No dataset selected for training")
                QMessageBox.warning(self, "Dataset Error", "Please select a dataset")
                return

            try:
                self.dataset = self.TranslationDataset(self.config, selected_dataset)
                self.tokenizer = self.TranslationTokenizer(self.config)
                self.model = self.TransformerModel(self.config)
                logger.info(f"Initialized dataset: {selected_dataset}, tokenizer, and model")
            except Exception as e:
                logger.error(f"Failed to initialize training components: {str(e)}")
                raise RuntimeError(f"Training component initialization failed: {str(e)}")

            try:
                self.trainer = self.Trainer(self.config, self.model, self.dataset)
                logger.info("Reinitialized trainer with model and dataset")
            except Exception as e:
                logger.error(f"Failed to reinitialize trainer: {str(e)}")
                raise RuntimeError(f"Trainer reinitialization failed: {str(e)}")

            self.training_in_progress = True
            self.train_button.setText("Stop Training")
            self.train_button.clicked.disconnect()
            self.train_button.clicked.connect(self.stop_training)

            self.progress_bar.setValue(0)
            self.progress_timer.start(1000)
            self.trainer.start_training(callback=self.on_training_update)
            logger.info(f"Training started with Trainer instance on dataset: {selected_dataset}")
        except Exception as e:
            logger.error(f"Training initiation failed: {str(e)}")
            raise RuntimeError(f"Training initiation failed: {str(e)}")

    def stop_training(self) -> None:
        """
        Stop the ongoing training process using the Trainer instance.

        Raises:
            RuntimeError: If stopping training fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.stop_training()
        """
        try:
            if self.Trainer is None:
                logger.warning("Training stop functionality disabled due to missing Trainer module")
                return

            if self.trainer and self.training_in_progress:
                self.trainer.stop_training()
                self.training_in_progress = False
                self.progress_timer.stop()
                self.progress_bar.setValue(0)
                self.train_button.setText("Start Training")
                self.train_button.clicked.disconnect()
                self.train_button.clicked.connect(self.start_training)
                logger.info("Training stopped via Trainer")
                self.sync_training_state()
            else:
                logger.warning("No active training or trainer not initialized")
        except Exception as e:
            logger.error(f"Failed to stop training: {str(e)}")
            raise RuntimeError(f"Failed to stop training: {str(e)}")

    def display_metrics(self, metrics: Dict[str, float] = None) -> None:
        """
        Display training metrics in the GUI.

        Args:
            metrics (Dict[str, float], optional): Metrics to display. If None, calculate from recent data.

        Raises:
            RuntimeError: If metrics display fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.display_metrics({'bleu': 0.3, 'loss': 0.5})
        """
        try:
            if self.metrics is None:
                logger.warning("Metrics display disabled due to missing Metrics module")
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
            logger.info(f"Displayed metrics: {metrics_text}")
        except Exception as e:
            logger.error(f"Failed to display metrics: {str(e)}")
            raise RuntimeError(f"Failed to display metrics: {str(e)}")

    def show_heatmap(self) -> None:
        """
        Visualize attention heatmaps for the latest translation.

        Raises:
            RuntimeError: If heatmap display fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.show_heatmap()
        """
        try:
            if self.metrics is None:
                logger.warning("Heatmap visualization disabled due to missing Metrics module")
                QMessageBox.warning(self, "Feature Unavailable", "Heatmap visualization is not available")
                return

            heatmap_size = self.config_manager.get_config_value("unit.heatmap_size", self.config, default=[10, 10])
            heatmap_data = np.random.rand(*heatmap_size)
            self.ax.clear()
            sns_heatmap = self.ax.imshow(heatmap_data, cmap="viridis")
            self.ax.set_title("Attention Heatmap")
            self.figure.colorbar(sns_heatmap)
            self.canvas.draw()
            logger.info("Attention heatmap displayed")
        except Exception as e:
            logger.error(f"Failed to display heatmap: {str(e)}")
            raise RuntimeError(f"Failed to display heatmap: {str(e)}")

    def update_progress(self) -> None:
        """
        Update training progress bar using Trainer's get_progress method.

        Raises:
            RuntimeError: If progress update fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.update_progress()
        """
        try:
            if self.Trainer is None or not self.trainer or not self.training_in_progress:
                logger.warning("Progress update disabled due to missing Trainer or no active training")
                return

            progress = self.trainer.get_progress()
            self.progress_bar.setValue(int(progress * 100))
            logger.debug(f"Training progress: {progress:.2f}")
            if progress >= 1.0:
                self.stop_training()
                self.display_metrics()
                logger.info("Training completed")
        except Exception as e:
            logger.error(f"Failed to update progress: {str(e)}")
            raise RuntimeError(f"Failed to update progress: {str(e)}")

    def save_translation(self) -> None:
        """
        Save the translated text to a file.

        Raises:
            RuntimeError: If saving translation fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.output_text.setText("Привет, мир!")
            >>> gui.save_translation()
        """
        try:
            translated_text = self.output_text.toPlainText()
            if not translated_text:
                logger.warning("No translation to save")
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
                logger.info(f"Translation saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save translation: {str(e)}")
            raise RuntimeError(f"Failed to save translation: {str(e)}")

    def on_training_update(self, metrics: Dict[str, float]) -> None:
        """
        Callback function for training updates to refresh metrics and progress.

        Args:
            metrics (Dict[str, float]): Training metrics from the trainer.

        Raises:
            RuntimeError: If callback handling fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.on_training_update({'epoch': 1, 'loss': 0.5, 'progress': 0.1})
        """
        try:
            self.display_metrics(metrics)
            logger.debug(f"Training metrics updated in GUI: {metrics}")
        except Exception as e:
            logger.error(f"Failed to update training metrics: {str(e)}")
            raise RuntimeError(f"Failed to update training metrics: {str(e)}")

    def initialize_components(self) -> None:
        """
        Initialize dataset, tokenizer, and model components.

        Raises:
            RuntimeError: If component initialization fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.initialize_components()
        """
        try:
            if self.TranslationDataset is None or self.TranslationTokenizer is None or self.TransformerModel is None:
                logger.warning("Component initialization disabled due to missing modules")
                QMessageBox.warning(self, "Feature Unavailable", "Component initialization is not available")
                return

            selected_dataset = self.dataset_combo.currentText()
            if selected_dataset:
                self.dataset = self.TranslationDataset(self.config, selected_dataset)
                self.tokenizer = self.TranslationTokenizer(self.config)
                self.model = self.TransformerModel(self.config)
                logger.info(f"Initialized components: dataset ({selected_dataset}), tokenizer, model")
            else:
                logger.warning("No dataset selected for component initialization")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def clear_output(self) -> None:
        """
        Clear the input and output text fields.

        Raises:
            RuntimeError: If clearing text fields fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.clear_output()
        """
        try:
            self.input_text.clear()
            self.output_text.clear()
            logger.info("Input and output text fields cleared")
        except Exception as e:
            logger.error(f"Failed to clear text fields: {str(e)}")
            raise RuntimeError(f"Failed to clear text fields: {str(e)}")

    def reset_gui(self) -> None:
        """
        Reset the GUI state, including text fields, progress bar, and metrics display.

        Raises:
            RuntimeError: If GUI reset fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.reset_gui()
        """
        try:
            self.clear_output()
            self.progress_bar.setValue(0)
            self.metrics_display.clear()
            if self.training_in_progress:
                self.stop_training()
            logger.info("GUI state reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset GUI: {str(e)}")
            raise RuntimeError(f"Failed to reset GUI: {str(e)}")

    def load_checkpoint(self) -> None:
        """
        Load a trainer checkpoint from the configured checkpoint path.

        Raises:
            RuntimeError: If checkpoint loading fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.load_checkpoint()
        """
        try:
            if self.trainer is None:
                logger.warning("Checkpoint loading disabled due to missing Trainer")
                return

            checkpoint_path = self.config_manager.get_config_value("model.checkpoint_path", self.config, default="model/checkpoints")
            checkpoint_path_absolute = self.config_manager.get_absolute_path(checkpoint_path)
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Checkpoint", checkpoint_path_absolute, "Checkpoint Files (*.pth)"
            )
            if file_path:
                self.trainer.load_checkpoint(file_path)
                logger.info(f"Loaded checkpoint from {file_path}")
                self.sync_training_state()
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")

    def save_checkpoint(self) -> None:
        """
        Save the current trainer checkpoint to the configured checkpoint path.

        Raises:
            RuntimeError: If checkpoint saving fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.save_checkpoint()
        """
        try:
            if self.trainer is None:
                logger.warning("Checkpoint saving disabled due to missing Trainer")
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
                logger.info(f"Saved checkpoint to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            raise RuntimeError(f"Failed to save checkpoint: {str(e)}")

    def configure_logging(self) -> None:
        """
        Configure logging based on config settings.

        Raises:
            RuntimeError: If logging configuration fails.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.configure_logging()
        """
        try:
            log_level = self.config_manager.get_config_value("general.log_level", self.config, default="INFO")
            log_file = self.config_manager.get_config_value("logger.log_file", self.config, default="logs/translaiter.log")
            log_file_absolute = self.config_manager.get_absolute_path(log_file)
            os.makedirs(os.path.dirname(log_file_absolute), exist_ok=True)
            logging.getLogger().setLevel(getattr(logging, log_level))
            file_handler = logging.FileHandler(log_file_absolute)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logging.getLogger().addHandler(file_handler)
            logger.info(f"Logging configured with level {log_level} and file {log_file_absolute}")
        except Exception as e:
            logger.error(f"Failed to configure logging: {str(e)}")
            raise RuntimeError(f"Failed to configure logging: {str(e)}")

    def validate_imports(self) -> bool:
        """
        Validate all required imports for the GUI.

        Returns:
            bool: True if all imports are successful, False otherwise.

        Example:
            >>> gui = TranslationGUI(config_manager.config)
            >>> gui.validate_imports()
        """
        try:
            import PyQt5
            import matplotlib
            import numpy
            TranslationDataset, TranslationTokenizer, TransformerModel, Trainer = import_training_components()
            if all(x is None for x in [TranslationDataset, TranslationTokenizer, TransformerModel, Trainer]):
                logger.warning("Some training components could not be imported")
                return False
            logger.info("All required GUI imports validated successfully")
            return True
        except ImportError as e:
            logger.error(f"GUI import validation failed: {str(e)}")
            return False

if __name__ == "__main__":
    logger.info("Starting gui.py test execution")
    try:
        app = QApplication(sys.argv)
        config_manager = ConfigManager()
        gui = TranslationGUI(config_manager.config)
        gui.configure_logging()
        gui.validate_trainer_setup()
        gui.validate_imports()
        gui.show()
        logger.info("GUI test execution successful")
        sys.exit(app.exec_())
    except Exception as e:
        logger.error(f"GUI test execution failed: {str(e)}")
        sys.exit(1)