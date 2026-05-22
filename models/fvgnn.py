import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class FiniteVolumeGraphNet(MessagePassing):
    """
    Graph Network for FV mesh surrogate.

    Message:  m_ij = f_msg( x_i || x_j || e_ij )
    Aggregate: M_i = sum_j m_ij
    Update:   x_i' = f_node( x_i || M_i )

    Args:
        n_f      : node feature size  (history + n_static_node_feat)
        e_f      : edge feature size  (10: d_vec, |d|, S_vec, |S|, skew, non_orth)
        msg_dim  : output size of the message function
        out_dim  : output size per node (1 if predicting scalar T)
        hidden   : width of all hidden layers
        aggr     : aggregation scheme — 'add' mirrors FV flux summation
        activation: nonlinearity
    """

    def __init__(
        self,
        n_f: int,
        e_f: int,
        msg_dim: int,
        out_dim: int,
        hidden: int = 128,
        aggr: str = 'add',
        activation=nn.ReLU,
    ):
        super().__init__(aggr=aggr)

        # message MLP: takes sender, receiver, and edge features
        self.msg_fnc = nn.Sequential(
            nn.Linear(2 * n_f + e_f, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, msg_dim),
        )

        # node update MLP: takes current node features + aggregated messages
        self.node_fnc = nn.Sequential(
            nn.Linear(n_f + msg_dim, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # x         : (N, n_f)
        # edge_index: (2, E)
        # edge_attr : (E, e_f)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i, x_j  : (E, n_f)  — receiver and sender features
        # edge_attr  : (E, e_f)
        tmp = torch.cat([x_i, x_j, edge_attr], dim=1)   # (E, 2*n_f + e_f)
        return self.msg_fnc(tmp)                          # (E, msg_dim)

    def update(self, aggr_out, x=None):
        # aggr_out: (N, msg_dim) — aggregated messages at each node
        tmp = torch.cat([x, aggr_out], dim=1)            # (N, n_f + msg_dim)
        return self.node_fnc(tmp)                         # (N, out_dim)


class FVSurrogate(nn.Module):
    """
    Stacked FiniteVolumeGraphNet layers.
    All intermediate layers operate in latent space (hidden_dim → hidden_dim).
    The final layer projects to out_dim (1 for scalar T).

    The input node features are first lifted to hidden_dim by a linear encoder,
    so every MP layer sees the same feature size.
    Edge features are encoded once and reused at every layer.
    """

    def __init__(
        self,
        in_node_feat: int,     # history + n_static_node_feat
        in_edge_feat: int,     # 10
        hidden_dim: int = 128,
        msg_dim: int = 128,
        out_dim: int = 1,
        n_mp_layers: int = 6,
        aggr: str = 'add',
    ):
        super().__init__()

        # Lift raw features to hidden_dim once
        self.node_encoder = nn.Linear(in_node_feat, hidden_dim)
        self.edge_encoder = nn.Linear(in_edge_feat, hidden_dim)

        # Intermediate MP layers: hidden_dim → hidden_dim
        self.mp_layers = nn.ModuleList([
            FiniteVolumeGraphNet(
                n_f=hidden_dim,
                e_f=hidden_dim,
                msg_dim=msg_dim,
                out_dim=hidden_dim,
                aggr=aggr,
            )
            for _ in range(n_mp_layers - 1)
        ])

        # Final MP layer: hidden_dim → out_dim
        self.mp_layers.append(
            FiniteVolumeGraphNet(
                n_f=hidden_dim,
                e_f=hidden_dim,
                msg_dim=msg_dim,
                out_dim=out_dim,
                aggr=aggr,
            )
        )

    def forward(self, data):
        x         = self.node_encoder(data.x)           # (N, hidden_dim)
        edge_attr = self.edge_encoder(data.edge_attr)   # (E, hidden_dim)

        for layer in self.mp_layers:
            x = layer(x, data.edge_index, edge_attr)    # (N, hidden_dim) → ... → (N, out_dim)

        return x.squeeze(-1)                            # (N,) for scalar T