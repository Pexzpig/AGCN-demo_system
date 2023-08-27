import json

from torch.utils.data import Dataset
import os
import numpy as np
import torch
from scipy.spatial.distance import cdist

def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0))
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D = D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class MyGCNDataset(Dataset):

    def __init__(self, data_path="data.json", split="train", not_attack_list=[1, 2, 3, 5, 6], attack_list=[4]):
        with open(data_path, "r") as f:
            data = json.load(f)

        self.data = {}
        for filename, v in data.items():
            graph_id = int(filename.split(".")[0].split("_")[1])

            flag = False
            for not_attack_id in not_attack_list:
                min_id = (not_attack_id - 1) * 100
                max_id = min_id + 99
                if graph_id >= min_id and graph_id <= max_id:
                    flag = True
                    v["label"] = 0
                    break

            for not_attack_id in attack_list:
                min_id = (not_attack_id - 1) * 100
                max_id = min_id + 99
                if graph_id >= min_id and graph_id <= max_id:
                    flag = True
                    v["label"] = 1
                    break

            if flag:
                self.data[filename] = v

        self.data = [x for x in self.data.items()]

        np.random.seed(0)
        total_num = len(self.data)
        total_indexes = np.arange(0, total_num)
        np.random.shuffle(total_indexes)
        if split == "train":
            self.indexes = total_indexes[:int(total_num * 0.7)]
        else:
            self.indexes = total_indexes[int(total_num * 0.7):]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        idx = self.indexes[idx]
        filename, graph_data = self.data[idx]

        data = graph_data["type"]
        A = graph_data["A"]
        label = graph_data["label"]

        data = torch.tensor(data)
        data = torch.nn.functional.one_hot(data, num_classes=34)
        A = torch.tensor(A)
        A = torch.sqrt(torch.sqrt(A))
        A = A / A.max()
        A = normalize(A)
        label = torch.tensor(label)

        return data.float(), A.float(), label.long()

class MyGCNPredictDataset(Dataset):

    def __init__(self, data_path="data.json"):
        with open(data_path, "r") as f:
            data = json.load(f)

        filename, graph_data = list(data.items())[0]

        self.data = {
            "type": torch.tensor(graph_data["type"]),
            "A": torch.tensor(graph_data["A"])
        }

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = self.data["type"]
        A = self.data["A"]

        data = torch.nn.functional.one_hot(data, num_classes=34)
        A = torch.sqrt(torch.sqrt(A))
        A = A / A.max()
        A = normalize(A)

        return data.float(), A.float()

if __name__ == "__main__":
    dataset = MyGCNDataset(split="predict")  # Here you use the split "predict"
    dataset[0]
