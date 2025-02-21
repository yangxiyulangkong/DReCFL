import copy
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
import math
import torch
import torch.nn as nn
from fl_library.algorithms.clients.client_DReCFL import client_DReCFL
from fl_library.algorithms.servers.serverbase import Server
from torch.utils.data import DataLoader

class DReCFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.global_model = None
        self.set_slow_clients()
        self.set_clients(client_DReCFL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget_server = []
        self.Budget_clustering = []
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        self.server_learning_rate = args.server_learning_rate
        self.init_threshold = 16
        self.threshold = 0
        self.head = self.clients[0].model.head
        self.opt_h = torch.optim.SGD(self.head.parameters(), lr=self.server_learning_rate)

    def train(self):
        s_t = time.time()
        self.send_models()

        for i in range(self.global_rounds+1):
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            if ((i == 10) & (self.num_new_clients > 0)):
                self.newhead = self.clients[0].model.head
                self.set_new_clients(client_DReCFL)
                self.new_client_send_models()
                self.clients.extend(self.new_clients)
                self.num_clients = self.num_clients + self.num_new_clients

            self.selected_clients = self.select_clients()

            for client in self.selected_clients:
                client.train()
                # client.getrawdata()
                client.collect_protos()
                # client.train()

            # self.receive_rawdata()
            # self.calculate_rawdata_dis()
            server_time = time.time()
            self.receive_protos()

            server_clustering = time.time()
            self.clustering()
            self.Budget_clustering.append(time.time() - server_clustering)

            # self.clustering_label()
            self.train_head()
            self.Budget_server.append(time.time()-server_time)
            print('server time', self.Budget_server[-1])

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            s_t = time.time()

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

    def send_models(self):

        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.head)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def new_client_send_models(self):
        assert (len(self.new_clients) > 0)

        for client in self.new_clients:
            start_time = time.time()

            client.set_parameters(self.newhead)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_rawdata(self):
        assert (len(self.selected_clients)>0)
        self.uploaded_ids = []
        self.uploaded_rawdatas = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for cc in client.rawdatas.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_rawdatas.append((client.id, client.rawdatas[cc], y))

    def calculate_rawdata_dis(self):
        c_id_list = []
        rawdata_list = []
        y_list = []

        rawdata_loader = DataLoader(self.uploaded_rawdatas, drop_last=False, shuffle=True)

        for c_id, rawdata_avg, y in rawdata_loader:
            c_id_list.append(c_id)
            rawdata_list.append(rawdata_avg)
            y_list.append(y)

        num_points = len(rawdata_list)
        print(num_points)
        for i in range(num_points):
            for j in range(num_points):
                # dis = wasserstein_distance(rawdata_list[i], rawdata_list[j])
                # dis = euclidean_distance(rawdata_list[i], rawdata_list[j])
                dis = cosine_distance(rawdata_list[i], rawdata_list[j])
                print(f"{y_list[i].data} {y_list[j].data} {dis}")

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for cc in client.protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                self.uploaded_protos.append((client.id, client.protos[cc], y))

    def clustering(self):
        c_id_list = []
        p_list = []
        y_list = []
        self.sample_clusters = []
        self.clu_max = 0

        proto_loader = DataLoader(self.uploaded_protos, drop_last=False, shuffle=True)

        for c_id, p, y in proto_loader:
            c_id_list.append(c_id)
            p_list.append(p)
            y_list.append(y)

        num_points = len(p_list)
        clusters = [-1] * num_points
        cluster_index = 0

        if (len(self.rs_train_loss)<2):
            ratio = self.rs_train_loss[-1] / self.rs_train_loss[0]
            threshold = 9.0 * (1 - math.cos(math.pi * (1 - ratio) / 2))
        else:
            ratio = (self.rs_train_loss[-1] / (self.rs_train_loss[-2])) * (self.rs_train_loss[-1] / (self.rs_train_loss[0]))
            threshold = 9.0 * (1 - math.cos(math.pi / 2 * (1 - ratio)))

        print(self.rs_train_loss[0])
        print(self.rs_train_loss[-1])
        print(threshold)

        for i in range(num_points):
            if clusters[i] == -1:
                clusters[i] = cluster_index
                ccluster_index = torch.tensor(cluster_index, dtype=torch.int64, device=self.device)
                self.sample_clusters.append((c_id_list[i], y_list[i], ccluster_index, p_list[i]))

                for j in range(i + 1, num_points):
                    # distance = wasserstein_distance(p_list[i], p_list[j])
                    # distance = euclidean_distance(p_list[i], p_list[j])
                    distance = cosine_distance(p_list[i], p_list[j])

                    if distance < threshold:
                        if clusters[j] == -1:
                            clusters[j] = cluster_index
                            ccluster_index = torch.tensor(cluster_index, dtype=torch.int64, device=self.device)
                            self.sample_clusters.append((c_id_list[j], y_list[j], ccluster_index, p_list[j]))
                cluster_index += 1
        self.clu_max = cluster_index


    def clustering_label(self):
        c_id_list = []
        p_list = []
        y_list = []
        self.sample_clusters = []
        self.clu_max = 0

        proto_loader = DataLoader(self.uploaded_protos, drop_last=False, shuffle=True)

        for c_id, p, y in proto_loader:
            c_id_list.append(c_id)
            p_list.append(p)
            y_list.append(y)

        num_points = len(p_list)
        clusters = [-1] * num_points
        cluster_index = 0

        for i in range(num_points):
            if clusters[i] == -1:
                clusters[i] = cluster_index
                ccluster_index = torch.tensor(cluster_index, dtype=torch.int64, device=self.device)
                self.sample_clusters.append((c_id_list[i], y_list[i], ccluster_index, p_list[i]))

                for j in range(i + 1, num_points):
                    if y_list[j] == y_list[i]:
                        if clusters[j] == -1:
                            clusters[j] = cluster_index
                            ccluster_index = torch.tensor(cluster_index, dtype=torch.int64, device=self.device)
                            self.sample_clusters.append((c_id_list[j], y_list[j], ccluster_index, p_list[j]))
                cluster_index += 1
        self.clu_max = cluster_index

    def train_head(self):
        id_list = []
        complete_head_list = []
        head_list = []

        self.head_copy = self.head
        proto_loader = self.sample_clusters

        for i in range(self.clu_max):
            self.head = self.head_copy
            for c_id, y, clu_id, p in proto_loader:
                if clu_id == i:
                    out = self.head(p)
                    loss = self.CEloss(out, y)
                    self.opt_h.zero_grad()
                    loss.backward()
                    self.opt_h.step()

            complete_head_list.append(copy.deepcopy(self.head))

        num_clients = len(self.selected_clients)
        num_classes = len(proto_loader) / num_clients
        num_classes = int(num_classes)

        i = 0
        total_head = 0
        sorted_proto_loader = sorted(proto_loader, key=lambda x: x[0])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for c_id, y, clu_id, p in sorted_proto_loader:
            i = i + 1
            total_head = total_head + complete_head_list[clu_id].weight.data
            if(i % num_classes == 0):
                avg_head = total_head/num_classes
                new_linear = nn.Linear(in_features=512, out_features=10, bias=True)
                new_linear.weight.data = nn.Parameter(avg_head)
                new_linear.to(device)
                head_list.append(new_linear)
                id_list.append(c_id)
                i = 0
                total_head = 0

        for client in self.selected_clients:
            start_time = time.time()
            k = 0
            for id in id_list:
                if client.id == id:
                    client.set_parameters(head_list[k])
                k = k+1

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

        total_head = 0
        for head_cid in head_list:
            total_head = total_head + head_cid.weight.data

        avg_head = total_head/num_clients
        new_linear = nn.Linear(in_features=512, out_features=10, bias=True)
        new_linear.weight.data = nn.Parameter(avg_head)
        new_linear.to(device)
        self.head = new_linear

# 1. Euclidean Distance
def euclidean_distance(data1, data2):
    return torch.sqrt(torch.sum((data1 - data2) ** 2))

# 2. Manhattan Distance or L1 Distance
def wasserstein_distance(data1, data2):
    return torch.sum(torch.abs(data1 - data2))

# 3. Cosine Distance
def cosine_distance(data1, data2):
    dot_product = torch.sum(data1 * data2)
    norm_p1 = torch.norm(data1)
    norm_p2 = torch.norm(data2)
    cosine_similarity = dot_product / (norm_p1 * norm_p2)
    cosine_similarity = torch.round(cosine_similarity * 10000) / 10000
    cosine_distance = cosine_similarity - 1
    cosine_distance = torch.abs(cosine_distance)
    return cosine_distance