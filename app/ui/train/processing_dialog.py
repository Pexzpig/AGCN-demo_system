import os
from PyQt5.QtCore import pyqtSignal, Qt, QThread
from PyQt5.QtWidgets import QVBoxLayout, QFileDialog, QPushButton, QLabel, QProgressBar, QDialog, QCheckBox, QHBoxLayout
from app.models.split_data import DataSplitter
from app.models.preprocess_data import DataPreprocessor

class FileWorker(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, file_path, save_dir, do_split=True):
        super().__init__()
        self.file_path = file_path
        self.save_dir = save_dir
        self.should_stop = False
        self.do_split = do_split

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            if self.do_split:
                # Run the data splitter
                self.data_splitter = DataSplitter(self.file_path, './resources/data/split')
                self.data_splitter.progress_signal.connect(self.progress_signal.emit) 
                self.data_splitter.split_data()

            # Run the data preprocessor
            self.data_preprocessor = DataPreprocessor('./resources/data/split', './resources/data/temp')
            self.data_preprocessor.progress_signal.connect(self.progress_signal.emit) 
            self.data_preprocessor.run()

        except Exception as e:
            print(f"An error occurred while saving file: {e}")


class DataProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('数据预处理')
        self.resize(600, 400)
        main_layout = QVBoxLayout()
        main_layout.addSpacing(50)

        # 上传文件布局
        upload_layout = QHBoxLayout()
        
        # 上传文件按钮
        self.upload_button = QPushButton('上传图数据(.tsv)')
        self.button_config(self.upload_button)
        self.upload_button.clicked.connect(self.upload_file)
        upload_layout.addWidget(self.upload_button)

        # 添加复选框
        self.split_checkbox = QCheckBox("进行图拆分")
        self.button_config(self.split_checkbox)
        self.split_checkbox.setChecked(True)
        upload_layout.addWidget(self.split_checkbox)

        main_layout.addLayout(upload_layout)

        main_layout.addSpacing(50)
        # 进度条
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        self.file_path_label = QLabel()
        main_layout.addWidget(self.file_path_label)

        self.setLayout(main_layout)
    
    def button_config(self, button, width=300, height=50, font_size=14):
        """配置按钮的大小和字体"""
        button.setFixedSize(width, height)
        font = button.font()
        font.setPointSize(font_size)
        button.setFont(font)

    def upload_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Open File', './resources/data', 'TSV Files (*.tsv)')
        if file_path:
            self.file_path_label.setText(f'已选文件: {file_path}\n\n数据处理中... 处理需要较长时间，请稍等')
            self.file_worker = FileWorker(file_path, './resources/data', self.split_checkbox.isChecked())
            self.file_worker.progress_signal.connect(self.progress_bar.setValue)
            self.file_worker.start()

    def closeEvent(self, event):
        if hasattr(self, 'file_worker') and self.file_worker.isRunning():
            self.file_worker.terminate()
        event.accept()
