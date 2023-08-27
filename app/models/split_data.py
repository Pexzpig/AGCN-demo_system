import os
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

class DataSplitter(QObject):
    progress_signal = pyqtSignal(int)

    def __init__(self, tsv_path="./resources/data/all.tsv", split_dir="./resources/data/split"):
        super(DataSplitter, self).__init__()
        self.tsv_path = tsv_path
        self.split_dir = split_dir

    def split_data(self):
        os.makedirs(self.split_dir, exist_ok=True)

        with open(self.tsv_path, "r") as f:
            lines = f.readlines()

        for index, line in enumerate(lines):
            items = line.split("\t")
            graph_id = int(items[-1])
            with open(os.path.join(self.split_dir, f"graph_{graph_id}.csv"), "a") as f:
                f.write(line)
                
            if graph_id % 12 == 0:
                self.progress_signal.emit(int(graph_id / 12))

        self.progress_signal.emit(50)