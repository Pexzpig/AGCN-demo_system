from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QHBoxLayout, QComboBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import json

class ParamsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('模型参数设置')
        self.resize(600, 400)
        
        # 加载当前的参数
        with open("./resources/settings.json", "r") as f:
            self.params = json.load(f)
        
        layout = QVBoxLayout()

        tooltips = {
            "Number of Epochs:": "训练轮次",
            "Seed:": "训练随机种子",
            "Dim Hidden:": "隐藏层维度",
            "Not Attack List (comma separated):": "数据集分割选择(默认使用全数据集)",
            "Learning Rate:": "学习率",
            "Weight Decay:": "防止过拟合的正则化参数",
            "Step Size:": "降低学习率所需的训练轮次",
            "Gamma:": "降低学习率的比率",
            "Save Directory:": "保存训练模型路径",
            "Data Path:": "已预处理的数据集路径",
            "Graph Size:": "图概要大小",
            "Resolution:": "Louvain算法分辨率",
            "Model Type:": "使用模型"
        }

        # 创建并初始化 QLineEdit 控件
        self.num_epochs_edit = self.create_horizontal_layout("Number of Epochs:", str(self.params["num_epochs"]), layout, tooltips)
        self.seed_edit = self.create_horizontal_layout("Seed:", str(self.params["seed"]), layout, tooltips)
        self.dim_hidden_edit = self.create_horizontal_layout("Dim Hidden:", str(self.params["dim_hidden"]), layout, tooltips)
        self.not_attack_list_edit = self.create_horizontal_layout("Not Attack List (comma separated):", ','.join(map(str, self.params["not_attack_list"])), layout, tooltips)
        self.lr_edit = self.create_horizontal_layout("Learning Rate:", str(self.params["lr"]), layout, tooltips)
        self.weight_decay_edit = self.create_horizontal_layout("Weight Decay:", str(self.params["weight_decay"]), layout, tooltips)
        self.step_size_edit = self.create_horizontal_layout("Step Size:", str(self.params["step_size"]), layout, tooltips)
        self.gamma_edit = self.create_horizontal_layout("Gamma:", str(self.params["gamma"]), layout, tooltips)
        self.graph_size_edit = self.create_horizontal_layout("Graph Size:", str(self.params.get("graph_size", 300)), layout, tooltips)
        self.resolution_edit = self.create_horizontal_layout("Resolution:", str(self.params.get("resolution", 4.0)), layout, tooltips)

        # 添加model type下拉列表
        layout_model_type = QHBoxLayout()
        label_model_type = QLabel("Model Type:")
        layout_model_type.addWidget(label_model_type)
        label_model_type.setToolTip(tooltips["Model Type:"])
        self.model_type_combobox = QComboBox()
        self.model_type_combobox.addItem("AttentionGCN", 1)
        self.model_type_combobox.addItem("GCN", 0)
        self.model_type_combobox.setCurrentIndex(self.model_type_combobox.findData(self.params.get("model_type", 1)))
        layout_model_type.addWidget(self.model_type_combobox)
        layout.addLayout(layout_model_type)
        
        # 文件路径选择
        self.save_dir_edit = self.create_horizontal_layout_with_button("Save Directory:", self.params["save_dir"], layout, "选择文件夹", self.choose_save_directory, tooltips)        
        self.data_path_edit = self.create_horizontal_layout_with_button("Data Path:", self.params["data_path"], layout, "选择文件", self.choose_data_file, tooltips)

        save_btn = QPushButton("保存修改", self)
        save_btn.clicked.connect(self.save_params)
        layout.addWidget(save_btn)

        self.setLayout(layout)

    def create_horizontal_layout(self, label_text, default_text, main_layout, tooltips):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setToolTip(tooltips[label_text])
        layout.addWidget(label, 1)
        edit = QLineEdit(default_text)
        layout.addWidget(edit, 1)
        main_layout.addLayout(layout)
        return edit

    def create_horizontal_layout_with_button(self, label_text, default_text, main_layout, btn_text, btn_function, tooltips):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setToolTip(tooltips[label_text])
        layout.addWidget(label)
        btn = QPushButton(btn_text)
        btn.clicked.connect(btn_function)
        layout.addWidget(btn)
        main_layout.addLayout(layout)
        edit = QLineEdit(default_text)
        main_layout.addWidget(edit)
        return edit

    def choose_save_directory(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Choose Save Directory")
        if dir_name:
            self.save_dir_edit.setText(dir_name)

    def choose_data_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Choose Data File", "", "JSON Files (*.json);;All Files (*)")
        if file_name:
            self.data_path_edit.setText(file_name)

    def save_params(self):
        self.params["num_epochs"] = int(self.num_epochs_edit.text())
        self.params["seed"] = int(self.seed_edit.text())
        self.params["dim_hidden"] = int(self.dim_hidden_edit.text())
        self.params["not_attack_list"] = [int(i) for i in self.not_attack_list_edit.text().split(',')]
        self.params["lr"] = float(self.lr_edit.text())
        self.params["weight_decay"] = float(self.weight_decay_edit.text())
        self.params["step_size"] = int(self.step_size_edit.text())
        self.params["gamma"] = float(self.gamma_edit.text())
        self.params["save_dir"] = self.save_dir_edit.text()
        self.params["data_path"] = self.data_path_edit.text()
        self.params["graph_size"] = int(self.graph_size_edit.text())
        self.params["resolution"] = float(self.resolution_edit.text())
        self.params["model_type"] = self.model_type_combobox.currentData()

        with open("./resources/settings.json", "w") as f:
            json.dump(self.params, f, indent=4)

        self.accept()
