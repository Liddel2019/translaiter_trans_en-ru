import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Translaiter: English to Russian")
        self.setGeometry(100, 100, 400, 300)

        # Основной виджет и layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Заголовок
        self.label = QLabel("Transformer Training Control", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)

        # Кнопка "Начать обучение"
        self.start_button = QPushButton("Start Training", self)
        self.start_button.clicked.connect(self.start_training)
        self.layout.addWidget(self.start_button)

        # Кнопка "Остановить обучение"
        self.stop_button = QPushButton("Stop Training", self)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)  # Изначально отключена
        self.layout.addWidget(self.stop_button)

        # Кнопка "Сохранить модель"
        self.save_button = QPushButton("Save Model", self)
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)  # Изначально отключена
        self.layout.addWidget(self.save_button)

        # Статус
        self.status_label = QLabel("Status: Idle", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

    def start_training(self):
        """Запускает процесс обучения (заглушка)."""
        self.status_label.setText("Status: Training...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        # Здесь будет вызов функции обучения из trainer.py
        print("Training started (placeholder).")

    def stop_training(self):
        """Останавливает обучение (заглушка)."""
        self.status_label.setText("Status: Stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(True)
        print("Training stopped (placeholder).")

    def save_model(self):
        """Сохраняет модель (заглушка)."""
        self.status_label.setText("Status: Model Saved")
        print("Model saved (placeholder).")

def run_gui():
    app = QApplication(sys.argv)
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()