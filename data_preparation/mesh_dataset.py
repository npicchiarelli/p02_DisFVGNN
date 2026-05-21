import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Tuple, Optional
from .normalization import FeatureNormalizer


class SingleMeshDataset(Dataset):
    """
    One simulation (one fixed mesh, one T sequence).

    Each sample:
        data.x      : (N, history + n_static)  — normalized T history
                       concatenated with static node geometry
        data.y      : (N,)                     — normalized T at t+history
        data.edge_attr : (E, F_e)              — normalized static edge features
    """

    def __init__(
        self,
        T_sequence: torch.Tensor,        # (n_steps, N)
        static_graph: Data,
        normalizer: FeatureNormalizer,
        history: int = 1,
        split_indices: Optional[torch.Tensor] = None,  # subset of timestep indices
    ):
        super().__init__()
        self.T = T_sequence
        self.graph = static_graph
        self.norm = normalizer
        self.history = history

        # Which (idx) values are valid samples?
        n_samples = len(T_sequence) - history
        if split_indices is not None:
            # filter to those that fit within bounds
            self.indices = split_indices[split_indices < n_samples]
        else:
            self.indices = torch.arange(n_samples)

        # Pre-normalise static features once
        self._norm_edge_attr = normalizer.transform_edge_attr(
            static_graph.edge_attr
        )
        self._norm_node_attr = normalizer.transform_node_attr(
            static_graph.node_attr
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int) -> Data:
        idx = self.indices[i].item()

        # (N, history) — normalized T window
        T_window = self.T[idx : idx + self.history]          # (history, N)
        T_window_norm = self.norm.transform_T(T_window)
        x_T = T_window_norm.T.contiguous()                   # (N, history)

        # Concatenate static node geometry: (N, history + n_static_node_feat)
        x = torch.cat([x_T, self._norm_node_attr], dim=1)

        # Target
        y = self.norm.transform_T(self.T[idx + self.history])  # (N,)

        data = Data(
            x=x,
            y=y,
            edge_index=self.graph.edge_index,
            edge_attr=self._norm_edge_attr,
            pos=self.graph.pos,
            num_nodes=self.graph.num_nodes,
        )
        return data
    
    def __repr__(self) -> str:
        N = self.graph.num_nodes
        E = self.graph.edge_index.shape[1]
        return (
            f"SingleMeshDataset("
            f"samples={len(self)}, "
            f"nodes={N}, "
            f"edges={E}, "
            f"history={self.history})"
        )


class MultiMeshDataset(Dataset):
    """
    Multiple simulations with different meshes (different BCs, geometries).
    Wraps a list of SingleMeshDataset objects with a flat index.
    """

    def __init__(self, datasets: List[SingleMeshDataset]):
        self.datasets = datasets
        # Build flat index: (dataset_idx, local_sample_idx)
        self._index: List[Tuple[int, int]] = []
        for d_idx, ds in enumerate(datasets):
            for s_idx in range(len(ds)):
                self._index.append((d_idx, s_idx))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i: int) -> Data:
        d_idx, s_idx = self._index[i]
        return self.datasets[d_idx][s_idx]
    
def temporal_split(
    T_sequence: torch.Tensor,
    static_graph: Data,
    normalizer: FeatureNormalizer,
    history: int = 1,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    # remainder → test
) -> Tuple[SingleMeshDataset, SingleMeshDataset, SingleMeshDataset]:

    n = len(T_sequence) - history
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    train_idx = torch.arange(0, n_train)
    val_idx   = torch.arange(n_train, n_train + n_val)
    test_idx  = torch.arange(n_train + n_val, n)

    # Fit normalizer on training data ONLY
    normalizer.fit(
        T_train=T_sequence[:n_train],
        edge_attr=static_graph.edge_attr,
        node_attr=static_graph.node_attr,
    )

    train_ds = SingleMeshDataset(T_sequence, static_graph, normalizer, history, train_idx)
    val_ds   = SingleMeshDataset(T_sequence, static_graph, normalizer, history, val_idx)
    test_ds  = SingleMeshDataset(T_sequence, static_graph, normalizer, history, test_idx)
    return train_ds, val_ds, test_ds
