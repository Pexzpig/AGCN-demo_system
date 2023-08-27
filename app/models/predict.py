import os
import json
import torch
import numpy as np
import networkx as nx
import community
import torch.nn.functional as F
import gc
from PyQt5.QtCore import QThread, pyqtSignal
from app.models.model import GCN, AttentionGCN
from app.models.dataset import MyGCNPredictDataset

class GraphAttackPredictor(QThread):
    output_signal = pyqtSignal(str)
    type_list = "abcdefghijklmnopqrstuvwxyzABCDEFGH"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, graph_path, model_path="./resources/default_epoch/epoch_best.pth", settings_path="./resources/settings.json"):
        super().__init__()
        self.graph_path = graph_path
        self.model_path = model_path

        with open(settings_path, 'r') as f:
            self.settings = json.load(f)
        
        self.resolution = self.settings.get("resolution", 4.0)
        self.num_communities = self.settings.get("graph_size", 300)

    def run(self):
        self.predict_attack(self.graph_path, self.model_path)

    def process_graph(self, file_path):
        data={}
        with open(file_path, "r") as f:
            lines = f.readlines()

            G = nx.Graph()
            node2type = {}

            for line in lines:
                items = line.split("\t")

                src_id = int(items[0])
                src_type = items[1]
                dst_id = int(items[2])
                dst_type = items[3]

                G.add_edge(src_id, dst_id)
                node2type[src_id] = self.type_list.find(src_type)
                node2type[dst_id] = self.type_list.find(dst_type)

            partition = community.best_partition(G, resolution=self.resolution)
            communities = set(partition.values())

            community_representatives = {community: max([node for node in partition.keys() if partition[node] == community], key=G.degree) for community in communities}

            sorted_communities = sorted(community_representatives.items(), key=lambda x: G.degree(x[1]), reverse=True)[:self.num_communities]
            community_representatives = {community: representative for community, representative in sorted_communities}

            community2type = {community: node2type[representative] for community, representative in community_representatives.items()}

            A = np.zeros((self.num_communities, self.num_communities)).astype("int")
            for line in lines:
                items = line.split("\t")

                src_id = int(items[0])
                dst_id = int(items[2])

                src_community = partition[src_id]
                dst_community = partition[dst_id]

                if src_community in community_representatives and dst_community in community_representatives:
                    A[list(community_representatives.keys()).index(src_community)][list(community_representatives.keys()).index(dst_community)] += 1

            A = A.tolist()

            data["graph_0.json"] = {
                "type": list(community2type.values()),
                "A": A
            }
            with open(os.path.join(f"temp_graph.json"), "w") as f:
                json.dump(data, f)
        
    def normalize(self, A, symmetric=True):
        A = A + torch.eye(A.size(0))
        d = A.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(A)

    def predict_attack(self, graph_path, model_path):

        self.process_graph(graph_path)

        dataset = MyGCNPredictDataset(data_path="temp_graph.json")
        data, A = dataset[0]
        data = data.unsqueeze(0).to(self.device)
        A = A.unsqueeze(0).to(self.device)

        dim_hidden = self.settings.get("dim_hidden", 16)
        model_type = self.settings.get("model_type", 1)

        if model_type == 0:
            trained_model = GCN(dim_in=34, dim_hidden=dim_hidden, dim_out=2)
        else:
            trained_model = AttentionGCN(dim_in=34, dim_hidden=dim_hidden, dim_out=2, attention_num=self.num_communities)
        
        trained_model.load_state_dict(torch.load(model_path))
        trained_model = trained_model.to(self.device)
        trained_model.eval()

        # Model inference
        with torch.no_grad():
            output = trained_model(data, A)

        # Parse the output
        predicted_class = torch.argmax(output).item()

        self.output_signal.emit(f"The graph is predicted as {'attack' if predicted_class == 1 else 'not attack'}")
        os.remove("temp_graph.json")
