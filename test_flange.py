"""
Standalone evaluation of trained flange FVGNN surrogates.

Flange counterpart of `test_parametric.py`: it mirrors the testing / rollout
section of `train.py` but loads *trained* models from disk, and additionally
reports the per-case prediction time against the `laplacianFoam` wall-clock
(parsed from `log.laplacianFoam`) to quantify the surrogate speed-up.

Unlike the parametric case (one graph per geometry, split over MESHES), the
flange is a SINGLE mesh split over TIME via `temporal_split`, so the test set is
the tail of the time sequence. `temporal_split` re-fits the normalizer on the
training timesteps deterministically, exactly as `train.py` does — no seed or
cached normalizer is needed to reproduce it.

By default it evaluates the two runs compared in `data_analysis.ipynb`:
`flange_history1_fv` and `flange_history1_nofv`. Each experiment's `history` and
FV / no-FV mode are inferred from its name:
    history<N>  → history = N
    *_nofv      → edge_attr sliced to the first 4 (geometry) features
    *_fv        → all 10 edge features (the FV features)

Examples
--------
    python test_flange.py                                  # h=1 fv + nofv, GPU
    FVGNN_CPU=1 python test_flange.py                      # same, on CPU
    FVGNN_EXPS=flange_history5_fv python test_flange.py    # a single h=5 run
"""

import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Data

from data_preparation.field import load_fields
from data_preparation.mesh_dataset import temporal_split
from data_preparation.normalization import FeatureNormalizer
from data_preparation.static_graph import build_static_graph
from export_results.saving_of import saving_of
from mesh2graph.utils import filter_of_time_directories
from models.fvgnn import FVSurrogate
from models.autoregressive_training import rollout

torch.default_dtype = torch.float32

# ── 0. Configuration (MUST match train.py) ──────────────────────────────────

case_name = "flange"
excluded_patches = ["patch1", "patch3"]
train_frac = 0.5
val_frac   = 0.15

# The runs loaded by data_analysis.ipynb.
exp_names = os.environ.get(
    "FVGNN_EXPS", "flange_history1_fv,flange_history1_nofv"
).split(",")

save_of_fields = os.environ.get("FVGNN_SAVE_OF", "1") == "1"

raw_data_dir = "../raw_data"
case_dir = os.path.join(raw_data_dir, case_name)
processed_data_dir = Path("../processed_data")

# FVGNN_CPU=1 forces CPU — e.g. to time the surrogate on the same hardware as
# the CPU-bound laplacianFoam solver for a like-for-like speed-up.
force_cpu = os.environ.get("FVGNN_CPU", "0") == "1"
device = torch.device("cpu") if force_cpu else torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 1. Load the (single) flange mesh once and reuse it for every run ────────

print(f"Loading {case_dir} ...")
static_graph_full = build_static_graph(case_dir, excluded_patches)
T_sequence = load_fields(case_dir, "T", excluded_patches=excluded_patches)
print(f"Static graph edge_attr: {tuple(static_graph_full.edge_attr.shape)}, "
      f"T sequence: {tuple(T_sequence.shape)}")


def sim_wallclock_seconds(cdir):
    """Total laplacianFoam wall-clock, from the last `ExecutionTime = <x> s`
    line of log.laplacianFoam. None if the log is missing/unparseable."""
    log_path = os.path.join(cdir, "log.laplacianFoam")
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


sim_t = sim_wallclock_seconds(case_dir)
all_times = filter_of_time_directories(case_dir)

summary = []

# ── 2. Evaluate each trained run ────────────────────────────────────────────

for exp_name in exp_names:
    exp_name = exp_name.strip()
    print(f"\n{'='*72}\n{exp_name}\n{'='*72}")

    m = re.search(r"history(\d+)", exp_name)
    if not m:
        print(f"[{exp_name}] cannot infer history from the name, skipping.")
        continue
    history = int(m.group(1))
    use_fv = not exp_name.endswith("_nofv")

    checkpoint_dir = processed_data_dir / exp_name / "checkpoints"
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"[{exp_name}] no model.pt at {model_path}, skipping.")
        continue

    pred_dir  = os.path.join(processed_data_dir, exp_name, "predictions")
    error_dir = os.path.join(processed_data_dir, exp_name, "errors")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(error_dir, exist_ok=True)

    # FV runs keep all 10 edge features; no-FV runs keep only the first 4
    # (pure geometry) — mirrors the commented-out slice in train.py.
    static_graph = static_graph_full.clone()
    if not use_fv:
        static_graph.edge_attr = static_graph.edge_attr[:, :4]
    print(f"history={history}, {'FV (10 edge feats)' if use_fv else 'no-FV (4 edge feats)'}")

    # temporal_split fits the normalizer on the training timesteps internally,
    # exactly as train.py does.
    normalizer = FeatureNormalizer()
    train_ds, val_ds, test_ds = temporal_split(
        T_sequence, static_graph, normalizer, history=history,
        train_frac=train_frac, val_frac=val_frac,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, "
          f"Test samples: {len(test_ds)}")
    if len(test_ds) == 0:
        print(f"[{exp_name}] no test samples, skipping.")
        continue

    in_node_feat = history + static_graph.node_attr.shape[1]
    in_edge_feat = static_graph.edge_attr.shape[1]

    model = FVSurrogate(
        in_node_feat=in_node_feat,
        in_edge_feat=in_edge_feat,
        hidden_dim=64,
        out_dim=1,
        n_mp_layers=1,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):.4e} "
          f"parameters from {model_path}")

    if save_of_fields:
        Path(Path(pred_dir)  / f'{exp_name}_predictions.foam').touch()
        Path(Path(error_dir) / f'{exp_name}_errors.foam').touch()
        for of_dir in ['system', 'constant']:
            shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(pred_dir, of_dir), dirs_exist_ok=True)
            shutil.copytree(os.path.join(case_dir, of_dir), os.path.join(error_dir, of_dir), dirs_exist_ok=True)

    # Warm-up so lazy CUDA init / autotune is not charged to the timing.
    with torch.no_grad():
        _ = model(test_ds[0].to(device))
    sync()

    # ── One-step (teacher-forced) inference, timed ──────────────────────────
    test_times = all_times[len(train_ds) + len(val_ds):-history]

    preds, trues = [], []
    sync()
    start = time.perf_counter()
    for i in range(len(test_ds)):
        batch = test_ds[i].to(device)
        with torch.no_grad():
            pred = model(batch)
        preds.append(pred.cpu())
        trues.append(batch.y.cpu())
    sync()
    step_time = time.perf_counter() - start

    T_pred_norm = torch.stack(preds, dim=0)
    T_true_norm = torch.stack(trues, dim=0)
    T_pred = test_ds.norm.inverse_transform_T(T_pred_norm)
    T_true = test_ds.norm.inverse_transform_T(T_true_norm)

    # ── Autoregressive rollout, timed ───────────────────────────────────────
    sync()
    start = time.perf_counter()
    T_pred_rollout, T_true_rollout = rollout(model, test_ds, device=device)
    sync()
    roll_time = time.perf_counter() - start

    # ── Pure `model(batch)` time (no __getitem__, no host<->device copies) ──
    fwd_time = time_pure_forward(model, test_ds)

    rollout_mae = torch.mean(torch.abs(T_pred_rollout - T_true_rollout), dim=1)
    np.save(os.path.join(checkpoint_dir, "rollout_mae.npy"), rollout_mae.cpu().numpy())

    mae = torch.mean(torch.abs(T_pred - T_true)).item()
    mse = torch.mean((T_pred - T_true) ** 2).item()

    # ── Timing / speed-up report ────────────────────────────────────────────
    n = len(test_ds)
    print(f"[{exp_name}] Test MAE: {mae:.6e} | Test MSE: {mse:.6e}")
    print(f"[{exp_name}] {n} steps | one-step: {step_time:.4f}s "
          f"({1e3*step_time/n:.3f} ms/step) | "
          f"rollout: {roll_time:.4f}s ({1e3*roll_time/n:.3f} ms/step)")
    print(f"[{exp_name}] pure model(batch): {fwd_time:.4f}s "
          f"({1e3*fwd_time/n:.3f} ms/step) — "
          f"{step_time/fwd_time:.1f}x of it is data prep + transfers")
    su_step = su_roll = su_fwd = None
    if sim_t is not None:
        su_step = sim_t / step_time if step_time > 0 else float("inf")
        su_roll = sim_t / roll_time if roll_time > 0 else float("inf")
        su_fwd  = sim_t / fwd_time  if fwd_time  > 0 else float("inf")
        print(f"[{exp_name}] laplacianFoam wall-clock: {sim_t:.4f}s | speed-up  "
              f"one-step: {su_step:.1f}x  rollout: {su_roll:.1f}x  "
              f"pure-fwd: {su_fwd:.1f}x")
    else:
        print(f"[{exp_name}] no log.laplacianFoam found — skipping speed-up.")

    summary.append((exp_name, mae, mse, step_time, roll_time, fwd_time,
                    su_step, su_roll, su_fwd, n))

    # ── OpenFOAM field export ───────────────────────────────────────────────
    if save_of_fields:
        print(f"[{exp_name}] Saving results to OpenFOAM fields...")
        saver = saving_of([".", "-case", case_dir])
        for i, t in enumerate(test_times):
            os.makedirs(os.path.join(pred_dir, t), exist_ok=True)
            os.makedirs(os.path.join(error_dir, t), exist_ok=True)

            saver.setScalarField(T_pred[i].numpy(), [0, 0, 0, 1, 0, 0, 0])
            saver.exportScalarField(t, pred_dir, "T")
            saver.setScalarField(T_pred_rollout[i].numpy(), [0, 0, 0, 1, 0, 0, 0])
            saver.exportScalarField(t, pred_dir, "T_r")
            saver.setScalarField(torch.abs(T_pred[i] - T_true[i]).numpy(), [0, 0, 0, 1, 0, 0, 0])
            saver.exportScalarField(t, error_dir, "T")
            saver.setScalarField(torch.abs(T_pred_rollout[i] - T_true_rollout[i]).numpy(), [0, 0, 0, 1, 0, 0, 0])
            saver.exportScalarField(t, error_dir, "T_r")


# ── 3. Summary ──────────────────────────────────────────────────────────────
if summary:
    print(f"\n{'='*72}\nSummary ({device}, laplacianFoam = "
          f"{sim_t:.2f}s)\n{'='*72}")
    print(f"{'run':<24} {'MAE':>11} {'e2e ms/step':>12} {'pure ms/step':>13} "
          f"{'e2e su':>8} {'pure su':>8}")
    for name, mae, mse, st, rt, ft, sus, sur, suf, n in summary:
        e2e = f"{sur:.1f}x" if sur is not None else "n/a"
        pure = f"{suf:.1f}x" if suf is not None else "n/a"
        print(f"{name:<24} {mae:>11.4e} {1e3*st/n:>12.2f} {1e3*ft/n:>13.3f} "
              f"{e2e:>8} {pure:>8}")
