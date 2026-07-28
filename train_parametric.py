import os
from pathlib import Path
import shutil
from glob import glob
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from data_preparation.field import load_fields
from data_preparation.mesh_dataset import SingleMeshDataset, MultiMeshDataset
from data_preparation.normalization import FeatureNormalizer
from data_preparation.static_graph import build_static_graph
from export_results.saving_of import saving_of
from mesh2graph.utils import filter_of_time_directories
from models.fvgnn import FVSurrogate
from soap import SOAP
from models.autoregressive_training import rollout

torch.default_dtype = torch.float32

# ── 0. Configuration ──────────────────────────────────────────────────

case_name = "parametric"
# The parametric meshes use zeroGradient on (top, bottom, cbores) — the
# analogue of the flange's excluded patch1/patch3 — and fixedValue on
# (sides, holes), which become the boundary nodes of the graph.
excluded_patches = ["top", "bottom", "cbores"]
epochs = 200
history = 1              # number of past timesteps used to predict the next
use_fv_features = True   # False → keep only the first 4 (geometry) edge features

# Derived, never hand-written: exp_name names the checkpoint directory and is
# parsed back by test_parametric.py, so it must always describe the run that
# actually ran. Building it from the switches above keeps the two in step.
exp_name = f"history{history}_mesh" + ("" if use_fv_features else "_nofv")

# The split is over MESHES, not over time: each mesh contributes its full
# time sequence to exactly one of train/val/test. The test meshes are
# geometries the network never sees during training, so the test error
# measures geometric adaptability.
train_mesh_frac = 0.6
val_mesh_frac   = 0.2
# remainder of the meshes → test
seed = 42

raw_data_dir = "../raw_data"
parametric_dir = os.path.join(raw_data_dir, case_name)
processed_data_dir = Path("../processed_data")
pdata_casename = f"{case_name}_{exp_name}"

os.makedirs(os.path.join(processed_data_dir, pdata_casename), exist_ok=True)

pred_dir  = os.path.join(processed_data_dir, pdata_casename, "predictions")
error_dir = os.path.join(processed_data_dir, pdata_casename, "errors")

os.makedirs(pred_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

checkpoint_dir = Path(processed_data_dir / pdata_casename / "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# Preprocessed meshes (static graph + T sequence) are cached here so that
# re-running the script skips the expensive OpenFOAM parsing — by far the
# slowest step. The cache is keyed by the excluded-patch set; delete this
# directory (or set use_cache=False) to force a rebuild after the raw data
# changes.
# use_cache = True
# preproc_cache_dir = processed_data_dir / "parametric_preproc_cache"
# os.makedirs(preproc_cache_dir, exist_ok=True)
# _excl_key = "-".join(excluded_patches) if excluded_patches else "none"


# def load_mesh(case_dir):
#     """Return (static_graph, T_sequence) for one case, caching to disk."""
#     name = os.path.basename(case_dir)
#     cache_file = preproc_cache_dir / f"{name}__T__{_excl_key}.pt"
#     if use_cache and cache_file.exists():
#         blob = torch.load(cache_file, weights_only=False)
#         return blob["graph"], blob["T"]
#     g = build_static_graph(case_dir, excluded_patches)
#     T = load_fields(case_dir, "T", excluded_patches=excluded_patches)
#     if use_cache:
#         torch.save({"graph": g, "T": T}, cache_file)
#     return g, T


# ── 1. Split the meshes into train / val / test ─────────────────────────────

case_dirs = sorted(glob(os.path.join(parametric_dir, "model_*")))
case_dirs = [d for d in case_dirs if os.path.isdir(d)]
n_meshes = len(case_dirs)
if n_meshes == 0:
    raise RuntimeError(f"No model_* cases found in {parametric_dir}")

perm = torch.randperm(n_meshes, generator=torch.Generator().manual_seed(seed)).tolist()
n_train = round(n_meshes * train_mesh_frac)
n_val   = round(n_meshes * val_mesh_frac)

train_mesh_idx = perm[:n_train]
val_mesh_idx   = perm[n_train:n_train + n_val]
test_mesh_idx  = perm[n_train + n_val:]

def names_of(idxs):
    return [os.path.basename(case_dirs[i]) for i in idxs]

print(f"Found {n_meshes} parametric meshes. Splitting over geometry:")
print(f"  train meshes ({len(train_mesh_idx)}): {names_of(train_mesh_idx)}")
print(f"  val   meshes ({len(val_mesh_idx)}): {names_of(val_mesh_idx)}")
print(f"  test  meshes ({len(test_mesh_idx)}): {names_of(test_mesh_idx)}")


# ── 2. Load every mesh's data ───────────────────────────────────────────────

static_graphs = []
T_sequences   = []
for case_dir in case_dirs:
    name = os.path.basename(case_dir)
    print(f"Loading {name} ...")
    g = build_static_graph(case_dir, excluded_patches)
    T = load_fields(case_dir, "T", excluded_patches=excluded_patches)
    if not use_fv_features:
        g.edge_attr = g.edge_attr[:, :4]  # geometry features only, drop the FV ones
    static_graphs.append(g)
    T_sequences.append(T)
    print(f"  static graph edge_attr: {tuple(g.edge_attr.shape)}, T sequence: {tuple(T.shape)}")


# ── 3. Fit a single normalizer on the TRAINING meshes only ──────────────────
# Statistics come exclusively from training geometries to avoid leaking
# anything about the held-out val/test meshes. Every SingleMeshDataset then
# shares this normalizer.

normalizer = FeatureNormalizer()
normalizer.fit(
    T_train=torch.cat([T_sequences[i].reshape(-1) for i in train_mesh_idx], dim=0),
    edge_attr=torch.cat([static_graphs[i].edge_attr for i in train_mesh_idx], dim=0),
    node_attr=torch.cat([static_graphs[i].node_attr for i in train_mesh_idx], dim=0),
)
normalizer.save(os.path.join(checkpoint_dir, "normalizer.pt"))

# ── 4. Build datasets ───────────────────────────────────────────────────────
# Each mesh uses its FULL time sequence (split_indices=None → all valid
# samples). Train/val meshes are wrapped into MultiMeshDatasets.

def make_ds(mesh_i):
    return SingleMeshDataset(
        T_sequences[mesh_i], static_graphs[mesh_i], normalizer, history,
    )

train_ds = MultiMeshDataset([make_ds(i) for i in train_mesh_idx])
val_ds   = MultiMeshDataset([make_ds(i) for i in val_mesh_idx])

n_test_samples = sum(len(T_sequences[i]) - history for i in test_mesh_idx)
print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, "
      f"Test samples: {n_test_samples}, "
      f"Total: {len(train_ds) + len(val_ds) + n_test_samples}")

# pin_memory speeds host→GPU copies; persistent_workers keeps the worker
# pool alive across all epochs instead of re-forking it every epoch.
num_workers = 4
loader_kwargs = dict(
    # Each sample is a FULL mesh, and parametric meshes vary ~6x in size
    # (up to ~336k edges). batch_size counts graphs, so a batch of large
    # meshes can reach several million edges; the message-passing MLP keeps
    # ~8 (E, hidden) activations for backprop, which OOMs a 16 GB GPU above
    # ~5M edges. batch_size=4 keeps the worst-case batch well under budget.
    batch_size=4,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=num_workers > 0,
)
train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

# ── 5. Model & optimiser ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Feature dimensions are identical across meshes; take them from the first.
in_node_feat  = history + static_graphs[0].node_attr.shape[1]  # T history + geometry
in_edge_feat  = static_graphs[0].edge_attr.shape[1]            # 10

model = FVSurrogate(
    in_node_feat=in_node_feat,
    in_edge_feat=in_edge_feat,
    hidden_dim=64,
    out_dim=1,
    n_mp_layers=1,         # message passing depth
).to(device)

print(f"Model initialized with parameters: {sum(p.numel() for p in model.parameters()):4e}")

optimizer = SOAP(model.parameters(), lr=3e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# ── 6. Training loop ────────────────────────────────────────────────────────
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
    train_loss = total_loss / len(train_loader)
    train_losses.append(train_loss)

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
    scheduler.step(val_loss)
    pbar.set_postfix({"train_loss": f"{train_loss:.4e}", "val_loss": f"{val_loss:.4e}"})

np.save(os.path.join(checkpoint_dir, "train_losses.npy"), train_losses)
np.save(os.path.join(checkpoint_dir, "val_losses.npy"), val_losses)
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

# ── 7. Testing and rollout on the held-out (unseen) meshes ──────────────────

model.eval()
all_mae, all_mse = [], []

for mesh_i in test_mesh_idx:
    case_dir = case_dirs[mesh_i]
    name = os.path.basename(case_dir)
    test_ds = make_ds(mesh_i)
    if len(test_ds) == 0:
        print(f"[{name}] no test samples, skipping.")
        continue

    # Per-mesh output directories (each mesh has its own geometry).
    case_pred_dir  = os.path.join(pred_dir, name)
    case_error_dir = os.path.join(error_dir, name)
    os.makedirs(case_pred_dir, exist_ok=True)
    os.makedirs(case_error_dir, exist_ok=True)

    # .foam placeholders so ParaView can open predictions and errors.
    Path(Path(case_pred_dir)  / f'{name}_predictions.foam').touch()
    Path(Path(case_error_dir) / f'{name}_errors.foam').touch()
    for of_dir in ['system', 'constant']:
        shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(case_pred_dir, of_dir), dirs_exist_ok=True)
        shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(case_error_dir, of_dir), dirs_exist_ok=True)

    preds, trues = [], []
    times = filter_of_time_directories(case_dir)
    test_times = times[:len(test_ds)]   # full sequence: window-start time per sample

    start_time = time.time()
    for i in range(len(test_ds)):
        batch = test_ds[i].to(device)
        with torch.no_grad():
            pred = model(batch)
        preds.append(pred.cpu())
        trues.append(batch.y.cpu())
    elapsed_time = time.time() - start_time
    print(f"[{name}] inference on {len(test_ds)} samples took {elapsed_time:.2f}s, "
          f"avg {elapsed_time/len(test_ds):.4f}s/sample")

    T_pred_norm = torch.stack(preds, dim=0)              # (n_test, N)
    T_true_norm = torch.stack(trues, dim=0)              # (n_test, N)

    T_pred = test_ds.norm.inverse_transform_T(T_pred_norm)
    T_true = test_ds.norm.inverse_transform_T(T_true_norm)

    T_pred_rollout, T_true_rollout = rollout(model, test_ds, device=device)

    rollout_mae = torch.mean(torch.abs(T_pred_rollout - T_true_rollout), dim=1)
    np.save(os.path.join(checkpoint_dir, f"rollout_mae_{name}.npy"), rollout_mae.cpu().numpy())

    mae = torch.mean(torch.abs(T_pred - T_true)).item()
    mse = torch.mean((T_pred - T_true) ** 2).item()
    all_mae.append(mae)
    all_mse.append(mse)
    print(f"[{name}] Test MAE: {mae:.6e} | Test MSE: {mse:.6e}")

    print(f"[{name}] Saving results to OpenFOAM fields...")
    saver = saving_of([".", "-case", case_dir])

    for i, t in enumerate(test_times):
        pred_time = os.path.join(case_pred_dir, t)
        err_time  = os.path.join(case_error_dir, t)
        os.makedirs(pred_time, exist_ok=True)
        os.makedirs(err_time, exist_ok=True)

        saver.setScalarField(T_pred[i].numpy(), [0, 0, 0, 1, 0, 0, 0])
        saver.exportScalarField(t, case_pred_dir, "T")
        saver.setScalarField(T_pred_rollout[i].numpy(), [0, 0, 0, 1, 0, 0, 0])
        saver.exportScalarField(t, case_pred_dir, "T_r")
        saver.setScalarField(torch.abs(T_pred[i] - T_true[i]).numpy(), [0, 0, 0, 1, 0, 0, 0])
        saver.exportScalarField(t, case_error_dir, "T")
        saver.setScalarField(torch.abs(T_pred_rollout[i] - T_true_rollout[i]).numpy(), [0, 0, 0, 1, 0, 0, 0])
        saver.exportScalarField(t, case_error_dir, "T_r")

if all_mae:
    print(f"\nMean Test MAE over {len(all_mae)} unseen meshes: {np.mean(all_mae):.6e}")
    print(f"Mean Test MSE over {len(all_mse)} unseen meshes: {np.mean(all_mse):.6e}")
