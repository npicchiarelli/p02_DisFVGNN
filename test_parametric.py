"""
Standalone evaluation of a trained parametric FVGNN surrogate.

This mirrors the testing / rollout section (everything after line ~232) of
`train_parametric.py`, but loads a *trained* model from disk instead of
training one. In addition to the per-mesh MAE/MSE and the OpenFOAM field
export, it measures the per-mesh GNN prediction time and compares it against
the `laplacianFoam` simulation wall-clock (parsed from `log.laplacianFoam`) to
report the surrogate speed-up.

The train/val/test split, the normalizer and the model architecture must match
`train_parametric.py` exactly, otherwise the loaded weights are evaluated on the
wrong meshes / wrong statistics. The relevant constants are duplicated below and
must be kept in sync with the training script.
"""

import os
from pathlib import Path
import re
import shutil
from glob import glob
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data

from data_preparation.field import load_fields
from data_preparation.mesh_dataset import SingleMeshDataset
from data_preparation.normalization import FeatureNormalizer
from data_preparation.static_graph import build_static_graph
from export_results.saving_of import saving_of
from mesh2graph.utils import filter_of_time_directories
from models.fvgnn import FVSurrogate
from models.autoregressive_training import rollout

torch.default_dtype = torch.float32

# ── 0. Configuration (MUST match train_parametric.py) ───────────────────────

case_name = "parametric"
# exp_name selects which trained checkpoint to evaluate and must match the
# history / edge-feature setup its model was trained with. Override with the
# FVGNN_EXP / FVGNN_HISTORY env vars, e.g. to evaluate the history=1 model:
#   FVGNN_EXP=history1_mesh_nofv FVGNN_HISTORY=1 python test_parametric.py
exp_name = os.environ.get("FVGNN_EXP", "history1_mesh")
excluded_patches = ["top", "bottom", "cbores"]
history = int(os.environ.get("FVGNN_HISTORY", "1"))
# train_parametric.py appends _nofv when it drops the FV edge features, so the
# checkpoint name alone tells us which edge-feature set the weights expect.
use_fv_features = not exp_name.endswith("_nofv")

train_mesh_frac = 0.6
val_mesh_frac   = 0.2
# remainder of the meshes → test
seed = 42

# Toggle the (slow) OpenFOAM field export off if you only care about the
# error metrics and the timing / speed-up numbers.
save_of_fields = False

raw_data_dir = "../raw_data"
parametric_dir = os.path.join(raw_data_dir, case_name)
processed_data_dir = Path("../processed_data")
pdata_casename = f"{case_name}_{exp_name}"

pred_dir  = os.path.join(processed_data_dir, pdata_casename, "predictions")
error_dir = os.path.join(processed_data_dir, pdata_casename, "errors")
checkpoint_dir = Path(processed_data_dir / pdata_casename / "checkpoints")

os.makedirs(pred_dir, exist_ok=True)
os.makedirs(error_dir, exist_ok=True)

model_path = os.path.join(checkpoint_dir, "model.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"No trained model at {model_path}. Run train_parametric.py first."
    )


# ── 1. Reproduce the exact mesh split ───────────────────────────────────────

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


print(f"Found {n_meshes} parametric meshes. Reproducing the geometry split:")
print(f"  train meshes ({len(train_mesh_idx)}): {names_of(train_mesh_idx)}")
print(f"  val   meshes ({len(val_mesh_idx)}): {names_of(val_mesh_idx)}")
print(f"  test  meshes ({len(test_mesh_idx)}): {names_of(test_mesh_idx)}")


# ── 2. Mesh loader (matches train_parametric.py, incl. edge_attr slicing) ───

def load_mesh(case_dir):
    """Return (static_graph, T_sequence) for one case, with the same edge
    features training used — all 10, or the first 4 (geometry) under _nofv."""
    g = build_static_graph(case_dir, excluded_patches)
    T = load_fields(case_dir, "T", excluded_patches=excluded_patches)
    if not use_fv_features:
        g.edge_attr = g.edge_attr[:, :4]
    return g, T


# ── 3. Get the normalizer ───────────────────────────────────────────────────
# train_parametric.py fits the normalizer on the TRAINING meshes only but does
# not save it. Load a cached normalizer if one exists; otherwise reconstruct it
# by re-fitting on the training meshes exactly as training does, then cache it.

normalizer = FeatureNormalizer()
norm_path = os.path.join(checkpoint_dir, "normalizer.pt")

if os.path.exists(norm_path):
    print(f"Loading normalizer from {norm_path}")
    normalizer.load(norm_path)
else:
    print("No cached normalizer found — re-fitting on the training meshes "
          "(this parses the training cases and may take a while)...")
    train_T, train_edge, train_node = [], [], []
    for i in train_mesh_idx:
        name = os.path.basename(case_dirs[i])
        print(f"  loading (train) {name} ...")
        g, T = load_mesh(case_dirs[i])
        train_T.append(T.reshape(-1))
        train_edge.append(g.edge_attr)
        train_node.append(g.node_attr)
    normalizer.fit(
        T_train=torch.cat(train_T, dim=0),
        edge_attr=torch.cat(train_edge, dim=0),
        node_attr=torch.cat(train_node, dim=0),
    )
    normalizer.save(norm_path)
    print(f"Saved normalizer to {norm_path}")


# ── 4. Load the test meshes ─────────────────────────────────────────────────

test_graphs = {}
test_T      = {}
for i in test_mesh_idx:
    name = os.path.basename(case_dirs[i])
    print(f"Loading (test) {name} ...")
    g, T = load_mesh(case_dirs[i])
    test_graphs[i] = g
    test_T[i]      = T
    print(f"  edge_attr: {tuple(g.edge_attr.shape)}, T sequence: {tuple(T.shape)}")


def make_ds(mesh_i):
    return SingleMeshDataset(
        test_T[mesh_i], test_graphs[mesh_i], normalizer, history,
    )


# ── 5. Rebuild the model and load the trained weights ───────────────────────

# FVGNN_CPU=1 forces CPU — e.g. to time the surrogate on the same hardware as
# the CPU-bound laplacianFoam solver for a like-for-like speed-up.
force_cpu = os.environ.get("FVGNN_CPU", "0") == "1"
device = torch.device("cpu") if force_cpu else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Feature dims are identical across meshes; take them from any test mesh.
any_graph = next(iter(test_graphs.values()))
in_node_feat = history + any_graph.node_attr.shape[1]   # T history + geometry
in_edge_feat = any_graph.edge_attr.shape[1]             # 10, or 4 under _nofv
model = FVSurrogate(
    in_node_feat=in_node_feat,
    in_edge_feat=in_edge_feat,
    hidden_dim=64,
    out_dim=1,
    n_mp_layers=1,
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model with {sum(p.numel() for p in model.parameters()):.4e} parameters "
      f"from {model_path}")

# ── 6. Simulation wall-clock helper ─────────────────────────────────────────

def sim_wallclock_seconds(case_dir):
    """Total laplacianFoam wall-clock time, from the last `ExecutionTime = <x> s`
    line of log.laplacianFoam. Returns None if the log is missing/unparseable."""
    log_path = os.path.join(case_dir, "log.laplacianFoam")
    if not os.path.exists(log_path):
        return None
    last = None
    pat = re.compile(r"ExecutionTime\s*=\s*([0-9.eE+-]+)\s*s")
    with open(log_path, "r", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                last = float(m.group(1))
    return last


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_pure_forward(model, ds, repeats=3):
    """Time ONLY `model(batch)`.

    The static graph and every input window are pre-staged on `device` up front,
    so the measurement excludes the SingleMeshDataset.__getitem__ work
    (normalisation, concat, Data construction) and every host<->device copy.
    edge_index / edge_attr are uploaded once and shared by all batches — they
    are identical across samples, so only `x` differs.

    Returns the best (minimum) total seconds over `repeats` passes.
    """
    edge_index_d = ds.graph.edge_index.to(device)
    edge_attr_d  = ds._norm_edge_attr.to(device)
    N = ds.graph.num_nodes

    batches = [
        Data(x=ds[i].x.to(device), edge_index=edge_index_d,
             edge_attr=edge_attr_d, num_nodes=N)
        for i in range(len(ds))
    ]

    best = float("inf")
    with torch.no_grad():
        model(batches[0])          # warm-up
        sync()
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            for b in batches:
                model(b)
            sync()
            best = min(best, time.perf_counter() - start)
    return best


# ── 7. Testing and rollout on the held-out (unseen) meshes ──────────────────

all_mae, all_mse = [], []
all_rmae, all_rmse = [], []
speedups_step, speedups_roll, speedups_fwd = [], [], []

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

    if save_of_fields:
        # .foam placeholders so ParaView can open predictions and errors.
        Path(Path(case_pred_dir)  / f'{name}_predictions.foam').touch()
        Path(Path(case_error_dir) / f'{name}_errors.foam').touch()
        for of_dir in ['system', 'constant']:
            shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(case_pred_dir, of_dir), dirs_exist_ok=True)
            shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(case_error_dir, of_dir), dirs_exist_ok=True)

    # One warm-up forward pass so lazy CUDA init / cuDNN autotune are not
    # charged to the measured inference time.
    with torch.no_grad():
        _ = model(test_ds[0].to(device))
    sync()

    # ── One-step (teacher-forced) inference, timed ──────────────────────────
    preds, trues = [], []
    times = filter_of_time_directories(case_dir)
    test_times = times[:len(test_ds)]   # full sequence: window-start time per sample

    sync()
    start_time = time.perf_counter()
    for i in range(len(test_ds)):
        batch = test_ds[i].to(device)
        with torch.no_grad():
            pred = model(batch)
        preds.append(pred.cpu())
        trues.append(batch.y.cpu())
    sync()
    step_time = time.perf_counter() - start_time

    T_pred_norm = torch.stack(preds, dim=0)              # (n_test, N)
    T_true_norm = torch.stack(trues, dim=0)              # (n_test, N)
    T_pred = test_ds.norm.inverse_transform_T(T_pred_norm)
    T_true = test_ds.norm.inverse_transform_T(T_true_norm)

    # ── Autoregressive rollout, timed ───────────────────────────────────────
    sync()
    start_time = time.perf_counter()
    T_pred_rollout, T_true_rollout = rollout(model, test_ds, device=device)
    sync()
    roll_time = time.perf_counter() - start_time

    # ── Pure `model(batch)` time (no __getitem__, no host<->device copies) ──
    fwd_time = time_pure_forward(model, test_ds)

    rollout_mae = torch.mean(torch.abs(T_pred_rollout - T_true_rollout), dim=1)
    np.save(os.path.join(checkpoint_dir, f"rollout_mae_{name}.npy"), rollout_mae.cpu().numpy())

    # ── Error metrics (one-step) ────────────────────────────────────────────
    mae = torch.mean(torch.abs(T_pred - T_true)).item()
    mse = torch.mean((T_pred - T_true) ** 2).item()
    rmae = (torch.sum(torch.abs(T_pred - T_true)) / torch.sum(torch.abs(T_true))).item()
    rmse = (torch.sum((T_pred - T_true) ** 2) / torch.sum(T_true ** 2)).item()
    all_mae.append(mae)
    all_mse.append(mse)
    all_rmae.append(rmae)
    all_rmse.append(rmse)

    # ── Timing / speed-up report ────────────────────────────────────────────
    n = len(test_ds)
    sim_t = sim_wallclock_seconds(case_dir)
    print(f"[{name}] Test MAE: {mae:.6e} | Test MSE: {mse:.6e}")
    print(f"[{name}] Test RMAE: {rmae:.6e} | Test RMSE: {rmse:.6e}")
    print(f"[{name}] {n} steps | one-step: {step_time:.4f}s "
          f"({1e3*step_time/n:.3f} ms/step) | "
          f"rollout: {roll_time:.4f}s ({1e3*roll_time/n:.3f} ms/step)")
    print(f"[{name}] pure model(batch): {fwd_time:.4f}s "
          f"({1e3*fwd_time/n:.3f} ms/step) — "
          f"{step_time/fwd_time:.1f}x of it is data prep + transfers")
    if sim_t is not None:
        su_step = sim_t / step_time if step_time > 0 else float("inf")
        su_roll = sim_t / roll_time if roll_time > 0 else float("inf")
        su_fwd  = sim_t / fwd_time  if fwd_time  > 0 else float("inf")
        speedups_step.append(su_step)
        speedups_roll.append(su_roll)
        speedups_fwd.append(su_fwd)
        print(f"[{name}] laplacianFoam wall-clock: {sim_t:.4f}s | speed-up  "
              f"one-step: {su_step:.1f}x  rollout: {su_roll:.1f}x  "
              f"pure-fwd: {su_fwd:.1f}x")
    else:
        print(f"[{name}] no log.laplacianFoam found — skipping speed-up.")

    # ── OpenFOAM field export ───────────────────────────────────────────────
    if save_of_fields:
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


# ── 8. Summary ──────────────────────────────────────────────────────────────
if all_mae:
    print(f"\nMean Test MAE over {len(all_mae)} unseen meshes: {np.mean(all_mae):.6e}")
    print(f"Mean Test MSE over {len(all_mse)} unseen meshes: {np.mean(all_mse):.6e}")
    print(f"Mean Test RMAE over {len(all_rmae)} unseen meshes: {np.mean(all_rmae):.6e}")
    print(f"Mean Test RMSE over {len(all_rmse)} unseen meshes: {np.mean(all_rmse):.6e}")
if speedups_roll:
    print(f"Mean speed-up vs laplacianFoam  —  one-step: "
          f"{np.mean(speedups_step):.1f}x  |  rollout: {np.mean(speedups_roll):.1f}x"
          f"  |  pure model(batch): {np.mean(speedups_fwd):.1f}x")
