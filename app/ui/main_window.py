from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from .train.train_window import TrainWindow
from .test_window import TestWindow
from .instructions_window import InstructionsWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('网络攻击检测')
        self.resize(1000, 600)
        self.open_main_window()

    def button_config(self, button, width=300, height=50, font_size=14):
        """配置按钮的大小和字体"""
        button.setFixedSize(width, height)
        font = button.font()
        font.setPointSize(font_size)
        button.setFont(font)

    def open_train_window(self):
        self.train_window = TrainWindow()
        self.train_window.back_signal.connect(self.open_main_window)
        self.setCentralWidget(self.train_window)

    def open_test_window(self):
        self.test_window = TestWindow()
        self.test_window.back_signal.connect(self.open_main_window)
        self.setCentralWidget(self.test_window)

    def open_instructions_window(self):
        self.instructions_window = InstructionsWindow()
        self.instructions_window.back_signal.connect(self.open_main_window)
        self.setCentralWidget(self.instructions_window)        

    def open_main_window(self):
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        title_label = QLabel("基于图深度学习网络攻击检测样例系统")
        title_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        title_label.setFont(font)
        layout.addWidget(title_label)

        train_button = QPushButton('训练数据')
        train_button.clicked.connect(self.open_train_window)
        self.button_config(train_button)
        layout.addWidget(train_button, alignment=Qt.AlignCenter)
        
        test_button = QPushButton('测试数据')
        test_button.clicked.connect(self.open_test_window)
        self.button_config(test_button)
        layout.addWidget(test_button, alignment=Qt.AlignCenter)

        instructions_button = QPushButton('系统说明')
        instructions_button.clicked.connect(self.open_instructions_window)
        self.button_config(instructions_button)
        layout.addWidget(instructions_button, alignment=Qt.AlignCenter)

        layout.addSpacing(50)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
