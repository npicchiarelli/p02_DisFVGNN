import os
from pathlib import Path
import shutil
from sys import exit
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_preparation.field import load_fields
from data_preparation.mesh_dataset import SingleMeshDataset, temporal_split
from data_preparation.normalization import FeatureNormalizer
from data_preparation.static_graph import build_static_graph
from export_results.saving_of import saving_of
from mesh2graph.utils import filter_of_time_directories
from models.fvgnn import FVSurrogate
from soap import SOAP
from models.autoregressive_training import rollout

torch.default_dtype = torch.float32

# ── 0. Configuration ──────────────────────────────────────────────────

case_name = "flange"
exp_name = "history5_time"
excluded_patches = ["patch1", "patch3"]
epochs = 200

raw_data_dir = "../raw_data"
case_dir = os.path.join(raw_data_dir, case_name)
processed_data_dir = Path("../processed_data")
pdata_casename = f"{case_name}_{exp_name}"

os.makedirs(os.path.join(processed_data_dir, pdata_casename), exist_ok=True)

pred_dir  = os.path.join(processed_data_dir, pdata_casename, "predictions")
error_dir = os.path.join(processed_data_dir, pdata_casename, "errors")

os.makedirs(pred_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

Path(Path(pred_dir)  / f'{pdata_casename}_predictions.foam').touch() # In order to visualize predictios and errors with paraview, we need a .foam placeholder
Path(Path(error_dir) / f'{pdata_casename}_errors.foam').touch()

for of_dir in ['system', 'constant']:
    shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(pred_dir, of_dir), dirs_exist_ok=True)
    shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(error_dir, of_dir), dirs_exist_ok=True)

checkpoint_dir = Path(processed_data_dir / pdata_casename / "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# ── 1. Load your mesh data ──────────────────────────────────────────────────

static_graph = build_static_graph(case_dir, excluded_patches)
T_sequence = load_fields(case_dir, 'T', excluded_patches=excluded_patches)
print("Static graph:", static_graph)
print("T sequence shape:", T_sequence.shape)  # Should be (T, N)

# ── 2. Split & normalise ────────────────────────────────────────────────────
normalizer = FeatureNormalizer()
history = 5   # use last 5 timesteps to predict next

train_ds, val_ds, test_ds = temporal_split(
    T_sequence, static_graph, normalizer, history=history,
    train_frac=0.5, val_frac=0.15
)

print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}, Total: {len(train_ds) + len(val_ds) + len(test_ds)}")
normalizer.save(os.path.join(checkpoint_dir, "normalizer.pt"))

train_loader = DataLoader(train_ds, batch_size=16, shuffle=False,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4)

# ── 3. Model & optimiser ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

in_node_feat  = history + static_graph.node_attr.shape[1]  # T history + geometry
in_edge_feat  = static_graph.edge_attr.shape[1]            # 10

model = FVSurrogate(
    in_node_feat=in_node_feat,
    in_edge_feat=in_edge_feat,
    hidden_dim=64,
    out_dim=1,
    n_mp_layers=1,         # message passing depth
).to(device)

print(f"Model initialized with parameters: {sum(p.numel() for p in model.parameters()):4e}")
print(f"Total training samples: {(len(train_ds))*len(static_graph.x):4e}")  # the first `history` timesteps only ever seed a window
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
optimizer = SOAP(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
# criterion = torch.nn.MSELoss()

# ── 4. Training loop ────────────────────────────────────────────────────────
train_losses = []
val_losses   = []

pbar = tqdm(range(epochs), desc="Training")
for epoch in pbar:
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)                    # (batch_N,)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            val_loss += torch.nn.functional.mse_loss(pred, batch.y).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    # scheduler.step(val_loss)
    pbar.set_postfix({"train_loss": f"{total_loss/len(train_loader):.4e}", "val_loss": f"{val_loss:.4e}"})
    # print(f"Epoch {epoch:4d} | train {total_loss/len(train_loader):.4e} | val {val_loss:.4e}")

np.save(os.path.join(checkpoint_dir, "train_losses.npy"), train_losses)
np.save(os.path.join(checkpoint_dir, "val_losses.npy"), val_losses)
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

# ── 5. Testing and rollout ────────────────────────────────────────────────────────

model.eval()
preds = []
trues = []
times = filter_of_time_directories(case_dir)
train_times = times[:len(train_ds)]
val_times = times[len(train_ds):len(train_ds)+len(val_ds)]
test_times = times[len(train_ds)+len(val_ds):-history]
# print(f"Testing on {len(test_times)} samples from times: {test_times}")
start_time = time.time()
for i in range(len(test_ds)):
    batch = test_ds[i].to(device)
    # print(f"Testing on sample {i+1}/{len(test_ds)} (time: {test_times[i]})")
    with torch.no_grad():
        pred = model(batch)
    preds.append(pred.cpu())
    trues.append(batch.y.cpu())

elapsed_time = time.time() - start_time
print(f"Inference on {len(test_ds)} samples took {elapsed_time:.2f} seconds, avg {elapsed_time/len(test_ds):.4f} seconds/sample")


T_pred_norm = torch.stack(preds, dim=0)              # (n_test, N)
T_true_norm = torch.stack(trues, dim=0)              # (n_test, N)

T_pred = test_ds.norm.inverse_transform_T(T_pred_norm)
T_true = test_ds.norm.inverse_transform_T(T_true_norm)

T_pred_rollout, T_true_rollout = rollout(model, test_ds, device=device)

rollout_mae = torch.mean(torch.abs(T_pred_rollout - T_true_rollout), dim = 1)

np.save(os.path.join(checkpoint_dir, "rollout_mae.npy"), rollout_mae.cpu().numpy())


print("Test MAE:", torch.mean(torch.abs(T_pred - T_true)).item())
print("Test MSE:", torch.mean((T_pred - T_true) ** 2).item())

print("Saving results to OpenFOAM  fields...")
saver = saving_of([".", "-case", case_dir])

# `t`, not `time`: the loop variable must not shadow the `time` module.
for i, t in enumerate(test_times):
    print(f"Exporting time {t} ({i+1}/{len(test_times)})")
    pred_time = os.path.join(pred_dir, t)
    err_time  = os.path.join(error_dir,  t)
    os.makedirs(pred_time, exist_ok=True)
    os.makedirs(err_time , exist_ok=True)

    saver.setScalarField(T_pred[i].numpy(), [0, 0, 0, 1, 0, 0, 0])
    saver.exportScalarField(t, pred_dir, "T")
    saver.setScalarField(T_pred_rollout[i].numpy(), [0, 0, 0, 1, 0, 0, 0])
    saver.exportScalarField(t, pred_dir, "T_r")
    saver.setScalarField(torch.abs(T_pred[i] - T_true[i]).numpy(), [0, 0, 0, 1, 0, 0, 0])
    saver.exportScalarField(t, error_dir, "T")
    saver.setScalarField(torch.abs(T_pred_rollout[i] - T_true_rollout[i]).numpy(), [0, 0, 0, 1, 0, 0, 0])
    saver.exportScalarField(t, error_dir, "T_r")
