from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit, QWidget, QDialog, QProgressBar, QApplication
from .processing_dialog import DataProcessingDialog
from .params_dialog import ParamsDialog
from .training_dialog import TrainingDialog


class TrainWindow(QWidget):
    back_signal = pyqtSignal()

    def button_config(self, button, width=300, height=50, font_size=14):
        """配置按钮的大小和字体"""
        button.setFixedSize(width, height)
        font = button.font()
        font.setPointSize(font_size)
        button.setFont(font)

    def __init__(self):
        super().__init__()

        # 主布局
        main_layout = QVBoxLayout()

        self.data_processing_button = QPushButton('数据预处理')
        self.button_config(self.data_processing_button)
        self.data_processing_button.clicked.connect(self.data_processing)
        main_layout.addWidget(self.data_processing_button, alignment=Qt.AlignCenter)

        self.set_params_button = QPushButton('设置模型参数')
        self.button_config(self.set_params_button)
        self.set_params_button.clicked.connect(self.set_model_params)
        main_layout.addWidget(self.set_params_button, alignment=Qt.AlignCenter)

        self.train_button = QPushButton('训练模型')
        self.button_config(self.train_button)
        self.train_button.clicked.connect(self.train_model)
        main_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        self.back_button = QPushButton('返回主页')
        self.back_button.clicked.connect(self.back_signal.emit)
        self.button_config(self.back_button)
        main_layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

    def data_processing(self):
        data_processing_dialog = DataProcessingDialog(self)
        data_processing_dialog.exec_()

    def set_model_params(self):
        params_dialog = ParamsDialog(self)
        params_dialog.exec_()

    def train_model(self):
        training_dialog = TrainingDialog(self)
        training_dialog.exec_()
