import torch
from data_preparation.mesh_dataset import SingleMeshDataset
from torch_geometric.data import Data


def train_step_autoregressive(
    model,
    batch,
    optimizer,
    history: int,
    k: int,                         # number of steps to unroll beyond the seed
    device: torch.device,
) -> float:
    """
    Unrolls the model for k steps. The window is seeded with `history`
    ground truth steps, then slides forward using model predictions.

    Step 0: window = [GT_0, GT_1, ..., GT_{h-1}]        → predict T_h
    Step 1: window = [GT_1, GT_2, ..., GT_{h-1}, T_h]   → predict T_{h+1}
    Step 2: window = [GT_2, ..., GT_{h-1}, T_h, T_{h+1}]→ predict T_{h+2}
    ...
    Step h: window = [T_h, T_{h+1}, ..., T_{h+k-1}]     → fully autoregressive
    """
    model.train()
    optimizer.zero_grad()

    n_static    = batch.x.shape[1] - history
    window      = batch.x[:, :history].clone()           # (batch_N, history) — GT seed
    node_static = batch.x[:, history:]                   # (batch_N, n_static) — shared

    total_loss = torch.tensor(0.0, device=device)

    for step in range(k):
        x = torch.cat([window.detach(), node_static], dim=1)      # (batch_N, history + n_static)

        data = Data(
            x=x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            num_nodes=x.shape[0],
        )

        T_next = model(data)                             # (batch_N,)
        target = batch.y[step]                        # (batch_N,) — see note below
        loss   = torch.nn.functional.mse_loss(T_next, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Slide window: drop oldest, append prediction (no teacher forcing)
        window = torch.roll(window, shifts=-1, dims=1)
        window[:, -1] = T_next.detach()

    total_loss /= k

    return total_loss.item()

def validation_step_autoregressive(
    model,
    batch,
    optimizer,
    history: int,
    k: int,                         # number of steps to unroll beyond the seed
    device: torch.device,
) -> float:
    """
    validation step for autoregressive training.
    """
    model.eval()

    n_static    = batch.x.shape[1] - history
    window      = batch.x[:, :history].clone()           # (batch_N, history) — GT seed
    node_static = batch.x[:, history:]                   # (batch_N, n_static) — shared

    val_loss = torch.tensor(0.0, device=device)

    for step in range(k):
        x = torch.cat([window, node_static], dim=1)      # (batch_N, history + n_static)

        data = Data(
            x=x,
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
            num_nodes=x.shape[0],
        )

        T_next = model(data)                             # (batch_N,)
        target = batch.y[step]                        # (batch_N,) — see note below
        val_loss  += torch.nn.functional.mse_loss(T_next, target)

        # Slide window: drop oldest, append prediction (no teacher forcing)
        window = torch.roll(window, shifts=-1, dims=1)
        window[:, -1] = T_next.detach()

    val_loss /= k

    return val_loss.item()

@torch.no_grad()
def rollout(
    model,
    test_ds: SingleMeshDataset,
    device: torch.device = torch.device('cpu'),
) -> tuple:
    """
    Autoregressive rollout over the test set.
    The first `history` steps are seeded from ground truth,
    every subsequent step uses the model's own predictions.

    Returns:
        T_pred : (n_test, N) predictions in physical units
        T_true : (n_test, N) ground truth in physical units
    """
    model.eval()
    model.to(device)

    norm       = test_ds.norm
    history    = test_ds.history
    T_sequence = test_ds.T
    n_steps    = len(test_ds)

    edge_attr_norm = test_ds._norm_edge_attr.to(device)
    node_attr_norm = test_ds._norm_node_attr.to(device)
    edge_index     = test_ds.graph.edge_index.to(device)
    N              = test_ds.graph.num_nodes

    seed_start = int(test_ds.indices[0])
    window = norm.transform_T(
        T_sequence[seed_start : seed_start + history]
    ).T.clone().to(device)                               # (N, history)

    preds = []
    trues = []

    for i in range(n_steps):
        x = torch.cat([window, node_attr_norm], dim=1)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_norm,
            num_nodes=N,
        )

        T_next = model(data)                             # (N,)

        preds.append(T_next.cpu())
        trues.append(norm.transform_T(T_sequence[seed_start + history + i]).cpu())

        window = torch.roll(window, shifts=-1, dims=1)
        window[:, -1] = T_next

    T_pred = norm.inverse_transform_T(torch.stack(preds, dim=0))  # (n_test, N)
    T_true = norm.inverse_transform_T(torch.stack(trues, dim=0))  # (n_test, N)

    return T_pred, T_true