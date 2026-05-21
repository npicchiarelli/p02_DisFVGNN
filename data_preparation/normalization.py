import torch


class FeatureNormalizer:
    """
    Fits on a list of T tensors (training timesteps only).
    Call .transform() on every Data object before feeding to the model.
    """
    def __init__(self):
        self.T_mean = None
        self.T_std = None
        self.edge_mean = None
        self.edge_std = None
        self.node_mean = None
        self.node_std = None

    def fit(self, T_train: torch.Tensor, edge_attr: torch.Tensor,
            node_attr: torch.Tensor):
        """
        T_train   : (n_steps, N) — training timesteps only
        edge_attr : (E, F_e)    — static, fit once
        node_attr : (N, F_n)    — static, fit once
        """
        self.T_mean = T_train.mean()
        self.T_std  = T_train.std().clamp(min=1e-8)

        self.edge_mean = edge_attr.mean(dim=0)
        self.edge_std  = edge_attr.std(dim=0).clamp(min=1e-8)

        self.node_mean = node_attr.mean(dim=0)
        self.node_std  = node_attr.std(dim=0).clamp(min=1e-8)

    def transform_T(self, T: torch.Tensor) -> torch.Tensor:
        return (T - self.T_mean) / self.T_std

    def inverse_transform_T(self, T_norm: torch.Tensor) -> torch.Tensor:
        return T_norm * self.T_std + self.T_mean

    def transform_edge_attr(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return (edge_attr - self.edge_mean) / self.edge_std

    def transform_node_attr(self, node_attr: torch.Tensor) -> torch.Tensor:
        return (node_attr - self.node_mean) / self.node_std

    def save(self, path: str):
        torch.save({
            "T_mean": self.T_mean, "T_std": self.T_std,
            "edge_mean": self.edge_mean, "edge_std": self.edge_std,
            "node_mean": self.node_mean, "node_std": self.node_std,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path)
        self.T_mean = ckpt["T_mean"];  self.T_std = ckpt["T_std"]
        self.edge_mean = ckpt["edge_mean"]; self.edge_std = ckpt["edge_std"]
        self.node_mean = ckpt["node_mean"]; self.node_std = ckpt["node_std"]