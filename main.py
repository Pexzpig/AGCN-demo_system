import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from app.ui.main_window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    app.setStyleSheet("QToolTip { "
                    "color: #000000; "
                    "background-color: #f2f2f2; "
                    "border: 1px solid #999999; "
                    "padding: 5px; "
                    "border-radius: 3px; "
                    "}")

    palette = QPalette()
    palette.setColor(QPalette.ToolTipBase, QColor(242, 242, 242))
    palette.setColor(QPalette.ToolTipText, Qt.black)
    app.setPalette(palette)

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec_())
