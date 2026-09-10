"""Compare the learned message m_ij against the finite-volume face flux.

Extracts the per-edge messages from a trained flange checkpoint with
`models.message_probe.MessageProbe`, builds the reference FV flux over the same
edges with `fv_flux`, and reports how much of one is in the other.

Configuration mirrors train.py / test_flange.py (same case, same split, so the
normalizer refits identically and the test window is the same tail of the time
sequence).

Reported diagnostics
--------------------
  best r           the best single message channel's Pearson r against the
                   target. High means one coordinate of the 128-dim message is
                   essentially a rescaled copy of it.
  R^2 (m -> .)     least-squares fit of the target on ALL message channels —
                   the honest "is it in there at all" number, since the message
                   is free to encode it in any linear combination. Run against
                   three targets, because F factorises exactly as F = w * dT:
                     F  = w*dT   the flux itself
                     dT          the bare temperature difference T_j - T_i
                     w           the geometric weight DT|Sf|/(|d| cos)
                   a message that tracks dT but not w has learned the driving
                   difference without the FV weighting — a different thing from
                   not being flux-like at all.
  R^2 (F -> m)     the converse, per channel, averaged: how much of the message
                   the flux explains (vs. information the FV scheme lacks).
  antisym. energy  ||A||^2/(||S||^2+||A||^2) for the split m_ij = S + A across
                   the two halves of a face. A conservative flux gives exactly
                   1; this is what makes an FV scheme conservative, and the
                   message is under no such constraint.
  aggregated       the same at node level, sum_j m_ij vs sum_j F_ij — the
                   quantity that actually drives dT_i/dt.

All the R^2 above are LINEAR read-outs. The node MLP that consumes the message
is not linear, so a low R^2 bounds how directly flux-like the message is, not
whether the flux is recoverable from it at all.

Examples
--------
    python analyze_messages.py
    FVGNN_EXPS=flange_history5_fv python analyze_messages.py
    FVGNN_EXPS=flange_history1_fv,flange_history1_nofv python analyze_messages.py
"""

import os
from pathlib import Path
import re

import numpy as np
import torch
from torch_geometric.data import Data

from data_preparation.field import load_fields
from data_preparation.mesh_dataset import temporal_split
from data_preparation.normalization import FeatureNormalizer
from data_preparation.static_graph import build_static_graph
from models.fvgnn import FVSurrogate
from models.message_probe import MessageProbe, fv_flux, paired_edge_index

torch.default_dtype = torch.float32

# ── 0. Configuration (MUST match train.py) ──────────────────────────────────

case_name = "flange"
excluded_patches = ["patch1", "patch3"]
train_frac = 0.5
val_frac   = 0.15
DT = 4e-5                        # constant/transportProperties

exp_names = os.environ.get("FVGNN_EXPS", "flange_history1_fv").split(",")
# 20 test steps x 32k edges x 128 channels is already ~330 MB of messages;
# the statistics below are converged well before the full 210-step test window.
max_steps = int(os.environ.get("FVGNN_MAX_STEPS", "20"))  # 0 = all test steps
out_dir = os.environ.get("FVGNN_MSG_OUT", "")             # non-empty = save .npz

raw_data_dir = "../raw_data"
case_dir = os.path.join(raw_data_dir, case_name)
processed_data_dir = Path("../processed_data")

device = torch.device("cpu" if os.environ.get("FVGNN_CPU", "0") == "1"
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")


# ── helpers ─────────────────────────────────────────────────────────────────

def corr(a, b):
    """Pearson r between (..., n) `a` rows and a (n,) vector `b`."""
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean()
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm()
    return num / den.clamp(min=1e-30)


def r2_multi(X, y):
    """R^2 of the least-squares fit y ~ [X, 1]."""
    X, y = X.double(), y.double()
    X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype)], dim=1)
    beta = torch.linalg.lstsq(X, y[:, None]).solution
    resid = y[:, None] - X @ beta
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot.clamp(min=1e-30))


# ── 1. Mesh + fields, loaded once ───────────────────────────────────────────

print(f"Loading {case_dir} ...")
static_graph_full = build_static_graph(case_dir, excluded_patches)
T_sequence = load_fields(case_dir, "T", excluded_patches=excluded_patches)

edge_index_raw = static_graph_full.edge_index
edge_attr_raw  = static_graph_full.edge_attr          # RAW geometry, 10 cols
E = edge_index_raw.shape[1]
N = static_graph_full.num_nodes
n_int = int((static_graph_full.node_attr[:, 0] == 1.0).sum())
is_bnd_edge = (edge_index_raw[0] >= n_int) | (edge_index_raw[1] >= n_int)
print(f"nodes={N} (internal {n_int}), edges={E} "
      f"(boundary-touching {int(is_bnd_edge.sum())}), T {tuple(T_sequence.shape)}")

rev = paired_edge_index(edge_index_raw)
has_rev = rev >= 0
print(f"edges with a reverse partner: {int(has_rev.sum())}/{E}")


# ── 2. Per-experiment probe ─────────────────────────────────────────────────

for exp_name in exp_names:
    exp_name = exp_name.strip()
    print(f"\n{'='*72}\n{exp_name}\n{'='*72}")

    m = re.search(r"history(\d+)", exp_name)
    if not m:
        print(f"[{exp_name}] cannot infer history from the name, skipping.")
        continue
    history = int(m.group(1))
    use_fv = not exp_name.endswith("_nofv")

    model_path = processed_data_dir / exp_name / "checkpoints" / "model.pt"
    if not model_path.exists():
        print(f"[{exp_name}] no model.pt at {model_path}, skipping.")
        continue

    static_graph = static_graph_full.clone()
    if not use_fv:
        static_graph.edge_attr = static_graph.edge_attr[:, :4]

    normalizer = FeatureNormalizer()
    train_ds, val_ds, test_ds = temporal_split(
        T_sequence, static_graph, normalizer, history=history,
        train_frac=train_frac, val_frac=val_frac,
    )
    if len(test_ds) == 0:
        print(f"[{exp_name}] no test samples, skipping.")
        continue

    model = FVSurrogate(
        in_node_feat=history + static_graph.node_attr.shape[1],
        in_edge_feat=static_graph.edge_attr.shape[1],
        hidden_dim=64,
        out_dim=1,
        n_mp_layers=1,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 2a. run the test window with the probe attached ────────────────────
    n_steps = len(test_ds) if max_steps <= 0 else min(max_steps, len(test_ds))
    edge_index_d = test_ds.graph.edge_index.to(device)
    edge_attr_d  = test_ds._norm_edge_attr.to(device)

    probe = MessageProbe(model)
    with probe, torch.no_grad():
        for i in range(n_steps):
            model(Data(x=test_ds[i].x.to(device), edge_index=edge_index_d,
                       edge_attr=edge_attr_d, num_nodes=N))
    print(probe)

    msg  = probe.stacked_messages(0)                 # (n_steps, E, msg_dim) f32
    aggr = probe.stacked_aggregated(0)               # (n_steps, N, msg_dim) f32

    # ── 2b. the FV reference over the SAME edges and the same timesteps ────
    # test_ds[i] predicts T at absolute step (idx + history); the message that
    # produced it was built from the window ending at (idx + history - 1), so
    # that is the field the flux must be evaluated on.
    idx0 = int(test_ds.indices[0])
    T_at_msg = T_sequence[idx0 + history - 1 : idx0 + history - 1 + n_steps]
    F = fv_flux(edge_index_raw, edge_attr_raw, T_at_msg.double(), DT=DT)  # (n_steps, E)
    # the two factors of F, kept apart for the controls below
    src_r, dst_r = edge_index_raw[0], edge_index_raw[1]
    dT_flat = (T_at_msg.double()[:, src_r] - T_at_msg.double()[:, dst_r])   # (n_steps, E)
    w = (DT * edge_attr_raw[:, 7].double()
         / (edge_attr_raw[:, 3].double() * edge_attr_raw[:, 9].double().clamp(min=0.05)))

    # ── 2c. edge-level comparison ─────────────────────────────────────────
    # cos(Sf, d) close to 1 is where fv_flux is exact — see the accuracy table
    # in message_probe.fv_flux. Elsewhere a mismatch may be the reference's.
    near_orth = (~is_bnd_edge) & (edge_attr_raw[:, 9] > 0.99)
    for label, mask in (("all edges", torch.ones(E, dtype=torch.bool)),
                        ("internal only", ~is_bnd_edge),
                        ("internal, cos>0.99", near_orth)):
        m_flat = msg[:, mask, :].reshape(-1, msg.shape[-1]).double()
        F_flat = F[:, mask].reshape(-1)                        # (n_steps*Em,)

        print(f"\n  [{label}]  {m_flat.shape[0]} edge-samples, "
              f"F range [{F_flat.min():.3e}, {F_flat.max():.3e}]")

        # F = w * dT exactly. Regressing the message on dT and on w separately
        # says WHICH factor it picked up: a message that tracks dT but not w has
        # learned the temperature difference without the geometric weighting,
        # which is a very different failure from not being flux-like at all.
        for tgt_name, tgt in (("F  = w*dT", F_flat),
                              ("dT = T_j-T_i", dT_flat[:, mask].reshape(-1)),
                              ("w  = DT|Sf|/(|d|cos)", w[mask].repeat(n_steps))):
            r = corr(m_flat.T, tgt)                            # (C,)
            k = int(r.abs().argmax())
            print(f"    vs {tgt_name:22s} best r = {r[k]:+.4f} (ch {k:3d}), "
                  f"|r|>0.5: {int((r.abs() > 0.5).sum()):3d}/{len(r)}, "
                  f"R^2 (m -> .) = {r2_multi(m_flat, tgt):.4f}")
        r2_back = np.mean([r2_multi(F_flat[:, None], m_flat[:, c])
                           for c in range(m_flat.shape[1])])
        print(f"    R^2 (F -> m), mean over channels : {r2_back:.4f}")

    # ── 2d. antisymmetry: F_ij = -F_ji exactly; is m_ij? ──────────────────
    # Split each quantity into its symmetric and antisymmetric parts across the
    # two halves of a face:  x_ij = S + A,  S = (x_ij + x_ji)/2, A = (x_ij - x_ji)/2.
    # The antisymmetric ENERGY FRACTION ||A||^2 / (||S||^2 + ||A||^2) is 1 for a
    # conservative flux and 0 for a purely symmetric quantity. This is stricter
    # than a cosine: it counts a constant offset as symmetric, which it is.
    def anti_fraction(x):
        S = 0.5 * (x + x[:, rev])
        A = 0.5 * (x - x[:, rev])
        sa = A.pow(2).sum(dim=(0, 1), dtype=torch.float64)
        ss = S.pow(2).sum(dim=(0, 1), dtype=torch.float64)
        return sa / (sa + ss).clamp(min=1e-300)

    anti_F = float(anti_fraction(F[:, :, None])[0])
    anti_m = anti_fraction(msg)                     # (C,)
    print(f"\n  antisymmetric energy fraction  ||A||^2/(||S||^2+||A||^2):")
    print(f"    FV flux             : {anti_F:.6f}   (1 = conservative, exact)")
    print(f"    message per channel : mean {anti_m.mean():.4f}  "
          f"min {anti_m.min():.4f}  max {anti_m.max():.4f}")
    print(f"    channels > 0.9      : {int((anti_m > 0.9).sum())}/{len(anti_m)}")

    # ── 2e. node level: the quantity that drives dT/dt ────────────────────
    F_node = torch.zeros(n_steps, N, dtype=torch.double)
    F_node.index_add_(1, edge_index_raw[1], F)             # sum_j F_ij
    a_flat = aggr.reshape(-1, aggr.shape[-1]).double()
    Fn_flat = F_node.reshape(-1)
    r_n = corr(a_flat.T, Fn_flat)
    kn = int(r_n.abs().argmax())
    print(f"\n  aggregated (sum_j m_ij) vs net FV flux (sum_j F_ij):")
    print(f"    best single channel : r = {r_n[kn]:+.4f}  (channel {kn})")
    print(f"    R^2 (M -> netF)     : {r2_multi(a_flat, Fn_flat):.4f}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{exp_name}_messages.npz")
        np.savez_compressed(
            path,
            messages=msg.numpy(), aggregated=aggr.numpy(),
            fv_flux=F.float().numpy(), fv_flux_node=F_node.float().numpy(),
            edge_index=edge_index_raw.numpy(), reverse_edge=rev.numpy(),
            is_boundary_edge=is_bnd_edge.numpy(),
        )
        print(f"\n  saved -> {path}")
