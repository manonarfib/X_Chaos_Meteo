import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, InstanceNorm


class Tanh(nn.Module):
    def forward(self, x):
        return torch.tanh(x)


class GNN_Layer_External(MessagePassing):
    def __init__(self, in_dim, out_dim, hidden_dim, ex_in_dim):
        super(GNN_Layer_External, self).__init__(node_dim=-2, aggr='mean')
        self.ex_embed_net_1 = nn.Sequential(nn.Linear(ex_in_dim + 2, hidden_dim), Tanh())
        self.ex_embed_net_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Tanh())
        self.message_net_1 = nn.Sequential(nn.Linear(in_dim + hidden_dim + 2, hidden_dim), Tanh())
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Tanh())
        self.update_net_1 = nn.Sequential(nn.Linear(in_dim + hidden_dim, hidden_dim), Tanh())
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_dim, out_dim), Tanh())
        self.norm = InstanceNorm(out_dim)

    def forward(self, in_x, ex_x, in_pos, ex_pos, edge_index, batch):
        n_in_x = in_x.size(0)
        ex_x = self.ex_embed_net_1(torch.cat((ex_x, ex_pos), dim=1))
        ex_x = self.ex_embed_net_2(ex_x)
        x = torch.cat((in_x, ex_x), dim=0)
        pos = torch.cat((in_pos, ex_pos), dim=0)
        index_shift = torch.zeros_like(edge_index)
        index_shift[0] = index_shift[0] + n_in_x
        x = self.propagate(edge_index + index_shift, x=x, pos=pos)
        x = x[:n_in_x]
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, pos_i, pos_j):
        message = self.message_net_1(torch.cat((x_i, x_j, pos_i - pos_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        return x + update


class GridMPNN(nn.Module):
    def __init__(self, n_passing, n_node_features_m, n_node_features_e, n_out_features, hidden_dim=128):
        super(GridMPNN, self).__init__()
        self.n_passing = n_passing
        self.hidden_dim = hidden_dim
        self.n_node_features_m = n_node_features_m
        self.n_node_features_e = n_node_features_e
        self.n_out_features = n_out_features

        self.gnn_ex_1 = GNN_Layer_External(in_dim=hidden_dim, out_dim=hidden_dim, hidden_dim=hidden_dim, ex_in_dim=n_node_features_e)
        self.gnn_ex_2 = GNN_Layer_External(in_dim=hidden_dim, out_dim=hidden_dim, hidden_dim=hidden_dim, ex_in_dim=n_node_features_e)

        self.embedding_mlp = nn.Sequential(
            nn.Linear(n_node_features_m + 2, hidden_dim), Tanh(),
            nn.Linear(hidden_dim, hidden_dim), Tanh()
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), Tanh(),
            nn.Linear(hidden_dim, n_out_features)
        )

    def forward(self, nodes, node_pos, edge_index, ex_nodes=None, ex_pos=None, edge_index_ex=None, batch=None):
        in_x = self.embedding_mlp(torch.cat((nodes, node_pos), dim=-1))

        if ex_nodes is not None and ex_pos is not None and edge_index_ex is not None:
            in_x = self.gnn_ex_1(in_x, ex_nodes, node_pos, ex_pos, edge_index_ex, batch)

        for _ in range(self.n_passing):
            in_x = in_x  # Internal message passing placeholder

        if ex_nodes is not None and ex_pos is not None and edge_index_ex is not None:
            in_x = self.gnn_ex_2(in_x, ex_nodes, node_pos, ex_pos, edge_index_ex, batch)

        out = self.output_mlp(in_x)
        return out