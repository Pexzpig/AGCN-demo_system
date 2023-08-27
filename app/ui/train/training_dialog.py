from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QTextEdit, QDialog
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import os

from app.models.train import TrainingThread

class TrainingWorker(QThread):
    output_signal = pyqtSignal(str)

    def run(self):
        try:
            self.training_instance = TrainingThread()
            self.training_instance.output_signal.connect(self.output_signal.emit)
            self.training_instance.run()

        except Exception as e:
            self.output_signal.emit(f"An error occurred while training: {e}")


class TrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('模型训练')
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        self.text_output = QTextEdit(self)
        self.text_output.setReadOnly(True)
        self.text_output.setFixedSize(450, 300)
        layout.addWidget(self.text_output, alignment=Qt.AlignCenter)

        self.train_button = QPushButton("开始训练", self)
        self.button_config(self.train_button)
        self.train_button.clicked.connect(self.start_training)
        layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        self.train_thread = TrainingWorker()
        self.train_thread.output_signal.connect(self.update_output)

    def button_config(self, button, width=300, height=50, font_size=14):
        """配置按钮的大小和字体"""
        button.setFixedSize(width, height)
        font = button.font()
        font.setPointSize(font_size)
        button.setFont(font)

    def start_training(self):
        self.train_button.setEnabled(False)
        self.text_output.append("Starting training...")
        self.train_thread.start()

    def update_output(self, text):
        self.text_output.append(text)

    def closeEvent(self, event):
        self.train_thread.terminate()
        super().closeEvent(event)

