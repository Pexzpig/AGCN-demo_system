from PyQt5.QtCore import QThread, pyqtSignal
import community
import networkx as nx
import json
import os
import numpy as np
import gc
import shutil
from PyQt5.QtCore import pyqtSignal

class DataPreprocessor(QThread):
    progress_signal = pyqtSignal(int)

    def __init__(self, split_dir='./resources/data/split', temp_dir='./resources/data/temp', settings_path="./resources/settings.json"):
        super().__init__()

        with open(settings_path, 'r') as f:
            self.settings = json.load(f)

        self.split_dir = split_dir
        self.temp_dir = temp_dir
        self.final_path = self.settings.get("data_path", "./resources/data/data.json")
        self.resolution = self.settings.get("resolution", 4.0)
        self.num_communities = self.settings.get("graph_size", 300)
        self.type_list = 'abcdefghijklmnopqrstuvwxyzABCDEFGH'
        self.total_data = {}

    def run(self):
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        self.progress_signal.emit(50)
        file_list = sorted(os.listdir(self.split_dir))

        for i in range(0, len(file_list), 50):
            for filename in file_list[i:i+50]:
                file_path = os.path.join(self.split_dir, filename)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                    G = nx.Graph()
                    node2type = {}

                    for line in lines:
                        items = line.split('\t')

                        src_id = int(items[0])
                        src_type = items[1]
                        dst_id = int(items[2])
                        dst_type = items[3]
                        edge_type = items[4]
                        graph_id = int(items[-1])

                        G.add_edge(src_id, dst_id)
                        node2type[src_id] = self.type_list.find(src_type)
                        node2type[dst_id] = self.type_list.find(dst_type)

                    partition = community.best_partition(G, resolution=self.resolution)
                    communities = set(partition.values())

                    community_representatives = {community: max([node for node in partition.keys() if partition[node] == community], key=G.degree) for community in communities}

                    sorted_communities = sorted(community_representatives.items(), key=lambda x: G.degree(x[1]), reverse=True)[:self.num_communities]
                    community_representatives = {community: representative for community, representative in sorted_communities}

                    community2type = {community: node2type[representative] for community, representative in community_representatives.items()}

                    A = np.zeros((self.num_communities, self.num_communities)).astype('int')
                    for line in lines:
                        items = line.split('\t')

                        src_id = int(items[0])
                        dst_id = int(items[2])

                        src_community = partition[src_id]
                        dst_community = partition[dst_id]

                        if src_community in community_representatives and dst_community in community_representatives:
                            A[list(community_representatives.keys()).index(src_community)][list(community_representatives.keys()).index(dst_community)] += 1

                    A = A.tolist()

                    self.total_data[filename] = {
                        'type': list(community2type.values()),
                        'A': A
                    }

                    del G, node2type, partition, communities, community_representatives, community2type, A
                    gc.collect()

            self.progress_signal.emit((i//50 + 1) * 4 + 50)

            # After processing 50 files, save the total_data to disk and then clear it
            with open(os.path.join(self.temp_dir, f'data_{i//50}.json'), 'w') as f:
                json.dump(self.total_data, f)
            
            self.total_data.clear()
            gc.collect()

        # After processing all files, merge all temporary files into the final file
        self._merge_temp_files()
        self.progress_signal.emit(100)

    def _merge_temp_files(self):
        temp_files = sorted(os.listdir(self.temp_dir))

        with open(self.final_path, 'w') as final_file, open(os.path.join(self.temp_dir, temp_files[0]), 'r') as first_file:
            final_file.write(first_file.read()[:-1])

        for temp_filename in temp_files[1:-1]:
            with open(self.final_path, 'a') as final_file, open(os.path.join(self.temp_dir, temp_filename), 'r') as temp_file:
                final_file.write(', ')
                final_file.write(temp_file.read()[1:-1])

        with open(self.final_path, 'a') as final_file, open(os.path.join(self.temp_dir, temp_files[-1]), 'r') as last_file:
            final_file.write(', ')
            final_file.write(last_file.read()[1:])

        shutil.rmtree(self.temp_dir)