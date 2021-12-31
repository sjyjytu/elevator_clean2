import numpy as np
import torch


class GraphBuilder:
    def __init__(self, lift_num, floor_num, device):
        self.lift_num = lift_num
        self.floor_num = floor_num
        self.node_num = self.floor_num + self.lift_num + self.floor_num + self.lift_num
        self.half_node = self.floor_num + self.lift_num
        self.device = device

    def get_zero_adj_matrix(self):
        adj_matrix = torch.eye(self.node_num).to(self.device)
        # connect corresponding up and down elements.
        for i in range(self.half_node):
            adj_matrix[i][i + self.half_node] = 1
            adj_matrix[i + self.half_node][i] = 1
        return adj_matrix

    def update_adj_matrix(self, adj_matrix, elv_up_call, elv_down_call):
        for idx, cur_elv_call in enumerate(elv_up_call):
            for floor_idx in cur_elv_call:
                adj_matrix[floor_idx][self.floor_num + idx] = 1
                adj_matrix[self.floor_num + idx][floor_idx] = 1
        for idx, cur_elv_call in enumerate(elv_down_call):
            for floor_idx in cur_elv_call:
                adj_matrix[self.half_node + floor_idx][self.half_node + self.floor_num + idx] = 1
                adj_matrix[self.half_node + self.floor_num + idx][self.half_node + floor_idx] = 1
        return adj_matrix

    def get_zero_node_feature(self):
        feature_matrix = torch.zeros((self.node_num, 3)).to(self.device)
        return feature_matrix

    def update_node_feature(self, feature_matrix, upload_wait_nums, download_wait_nums, load_up, load_down, location):
        for i in range(self.node_num):
            if i < self.floor_num:  # up floor
                feature_matrix[i][0] = 1  # floor or elevator
                feature_matrix[i][1] = upload_wait_nums[i]
                feature_matrix[i][2] = i / self.floor_num
            elif i < self.half_node:  # up elevator
                feature_matrix[i][0] = -1
                feature_matrix[i][1] = load_up[i - self.floor_num]
                feature_matrix[i][2] = location[i - self.floor_num]
            elif i < self.half_node + self.floor_num:  # down floor
                feature_matrix[i][0] = 1
                feature_matrix[i][1] = download_wait_nums[i - self.half_node]
                feature_matrix[i][2] = (i - self.half_node) / self.floor_num
            else:  # down elevator
                feature_matrix[i][0] = -1
                feature_matrix[i][1] = load_down[i - self.half_node - self.floor_num]
                feature_matrix[i][2] = location[i - self.half_node - self.floor_num]
        return feature_matrix

    def get_graph_pool(self, device, graph_pool_type='average'):
        n_nodes = self.node_num
        # batch_size is the shape of batch
        # for graph pool sparse matrix
        if graph_pool_type == 'average':
            elem = torch.full(size=(1 * n_nodes, 1),
                              fill_value=1 / n_nodes,
                              dtype=torch.float32,
                              device=device).view(-1)
        else:
            elem = torch.full(size=(1 * n_nodes, 1),
                              fill_value=1,
                              dtype=torch.float32,
                              device=device).view(-1)
        idx_0 = torch.arange(start=0, end=1,
                             device=device,
                             dtype=torch.long)
        idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((1 * n_nodes, 1)).squeeze()

        idx_1 = torch.arange(start=0, end=n_nodes * 1,
                             device=device,
                             dtype=torch.long)
        idx = torch.stack((idx_0, idx_1))
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([1, n_nodes * 1])).to(device)
        return graph_pool.to(self.device)
