from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont

class InstructionsWindow(QWidget):
    back_signal = pyqtSignal()

    def button_config(self, button, width=300, height=50, font_size=14):
        """配置按钮的大小和字体"""
        button.setFixedSize(width, height)
        font = button.font()
        font.setPointSize(font_size)
        button.setFont(font)

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        instructions_text = """
<h1>网络攻击图检测系统</h1>

<p>本系统基于注意力图卷积网络，旨在检测和识别网络攻击图。系统提供模型的训练和识别应用功能。</p>

<h2>主要功能</h2>

<ol>
    <li><strong>训练数据板块</strong>：用于上述模型的训练。
        <ul>
            <li><strong>参数调整</strong>：用户可以通过设置模型参数按钮来调整用于预处理以及模型训练的参数。</li>
            <li><strong>初始图数据集录入</strong>：支持tsv格式单文件上传，数据格式包括：source-id、source-type、destination-id、destination-type、edge-type、graph-id。</li>
            <li><strong>预处理</strong>：数据集预处理分为数据集拆分与图概要两部分。</li>
        </ul>
    </li>
    <li><strong>资源文件夹</strong>：已训练的模型和处理过的数据集都保存在<code>resources</code>文件夹下。</li>
    <li><strong>测试数据板块</strong>：用于验证和使用已训练的模型。
        <ul>
            <li>用户可以选择一个图文件来测试该图是否为攻击图。图文件格式为csv，数据格式要求与初始图相同。</li>
        </ul>
    </li>
</ol>
        """

        instructions_label = QLabel(instructions_text)
        
        font = QFont("宋体", 12)
        instructions_label.setFont(font)
        
        instructions_label.setWordWrap(True)
        
        # 设置文本左对齐，并设置整体布局的左右边距
        instructions_label.setAlignment(Qt.AlignLeft)
        margin = int((self.width() - instructions_label.width()) / 2)
        layout.setContentsMargins(50, 30, 50, 30)
        
        layout.addWidget(instructions_label)

        back_button = QPushButton("返回主页")
        back_button.clicked.connect(self.back_signal.emit)
        self.button_config(back_button)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)
