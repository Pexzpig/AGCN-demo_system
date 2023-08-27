from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QVBoxLayout, QPushButton, QLabel, QLineEdit, QWidget, QDialog, QHBoxLayout
from app.models.predict import GraphAttackPredictor

class TestWindow(QWidget):
    back_signal = pyqtSignal()

    def button_config(self, button, width=300, height=50, font_size=14):
        """配置按钮的大小和字体"""
        button.setFixedSize(width, height)
        font = button.font()
        font.setPointSize(font_size)
        button.setFont(font)

    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout()

        main_layout.addSpacing(50)

        # Upload test graph button and path
        graph_layout = QHBoxLayout()
        self.upload_graph_button = QPushButton('上传测试图(.csv)')
        self.button_config(self.upload_graph_button)
        self.upload_graph_button.clicked.connect(self.upload_graph)
        graph_layout.addWidget(self.upload_graph_button)

        self.graph_path_edit = QLineEdit()
        self.graph_path_edit.setFixedSize(400, 50)
        self.graph_path_edit.setReadOnly(True)
        graph_layout.addWidget(self.graph_path_edit)
        main_layout.addLayout(graph_layout)

        # Choose model button and path
        model_layout = QHBoxLayout()
        self.choose_model_button = QPushButton('选择模型(.pth)')
        self.button_config(self.choose_model_button)
        self.choose_model_button.clicked.connect(self.choose_model)
        model_layout.addWidget(self.choose_model_button)

        self.model_path_edit = QLineEdit("./resources/default_epoch.pth")
        self.model_path_edit.setFixedSize(400, 50)
        self.model_path_edit.setReadOnly(True)
        model_layout.addWidget(self.model_path_edit)
        main_layout.addLayout(model_layout)

        # Start test button
        self.start_test_button = QPushButton('开始测试')
        self.button_config(self.start_test_button)
        self.start_test_button.clicked.connect(self.start_test)
        main_layout.addWidget(self.start_test_button, alignment=Qt.AlignCenter)

        # Output textbox
        self.output_edit = QLineEdit()
        self.output_edit.setFixedSize(550, 50)
        self.output_edit.setReadOnly(True)
        main_layout.addWidget(self.output_edit, alignment=Qt.AlignCenter)

        # Back button
        self.back_button = QPushButton('返回主页')
        self.button_config(self.back_button)
        self.back_button.clicked.connect(self.back_signal.emit)
        main_layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        self.predictor = GraphAttackPredictor("", "")
        self.predictor.output_signal.connect(self.update_output)

    def upload_graph(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', './resources', 'CSV Files (*.csv)')
        if file_path:
            self.graph_path_edit.setText(file_path)

    def choose_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, 'Choose Model', './resources', 'PyTorch Model Files (*.pth)')
        if model_path:
            self.model_path_edit.setText(model_path)

    def start_test(self):
        graph_path = self.graph_path_edit.text()
        model_path = self.model_path_edit.text()

        if not graph_path or not model_path:
            self.output_edit.setText("请先选择测试图和模型!")
            return

        self.predictor.graph_path = graph_path
        self.predictor.model_path = model_path
        self.predictor.start()

    def update_output(self, text):
        self.output_edit.setText(text)
