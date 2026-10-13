"""
Collect the seed runs of each multi-seed experiment into one file.

train_parametric.py run with FVGNN_SEED=<n> writes to
processed_data/parametric_<exp_name>/seed_<n>/. This stacks what those runs
saved along a leading seed axis and writes it next to them:

    processed_data/parametric_<exp_name>/seeds.npz

A rerun never overwrites: when seed_<n> is taken, train_parametric.py writes to
seed_<n>_01, then seed_<n>_02, ... Each rerun index is a separate set of seeds,
collected into its own file, seeds_01.npz, seeds_02.npz, ..., so two sweeps of
one experiment are never mixed. The index is the one test_parametric.py takes
as FVGNN_RUN; 0 is the un-numbered seed_<n> runs and seeds.npz.

Data only: no statistics and no plots, those are data_analysis.ipynb's job.

Keys, for S seeds, E epochs, C message channels and T_m rollout steps on test
mesh m:

    exp_name                  str
    seeds                     (S,)     the seed of each row, ascending
    test_meshes               (M,)     names of the test meshes
    n_params                  (S,)     parameter count of each model.pt
    train_losses, val_losses  (S, E)   per-epoch MSE (train: without penalty)
    lrs                       (S, E)   learning rate each epoch trained at
    msg_<key>                 (S, E) or (S, E, C), each per-epoch entry of
                              msg_reg_log.npz: pr, pen, sigma, ...
    msg_c0, msg_ref           (S,)     set during each run, so one per seed
    msg_kind, msg_alpha, msg_warmup, msg_ramp_len, msg_ema
                              one value, the same for every seed
    rollout_mae_<mesh>        (S, T_m) autoregressive rollout MAE per step

A seed enters once its run has saved all of the above; unfinished ones are
listed and left out, so every array shares the same seed axis. The seeds of
one experiment must agree on everything that is not per seed (epochs, test
meshes, parameter count, message-penalty settings), otherwise the experiment
is reported and its seeds.npz left as it was. An unseeded run at the top of
an experiment directory is not a seed and is ignored.

    python aggregate_seeds.py                # every experiment with seed_<n> runs
    python aggregate_seeds.py <exp_name> ... # only these, every rerun index
    python aggregate_seeds.py <exp_name>:<i> # only rerun i of it (0: seeds.npz)
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

from models.msg_regularization import MessageRegularizer

case_name = "parametric"
processed_data_dir = Path("../processed_data")

# Set during each run: c0 from the initial weights, l1's ref over the warmup.
MSG_PER_SEED = ("c0", "ref")


class Inconsistent(Exception):
    """The seeds of one experiment cannot be stacked."""


def seed_dirs(exp_dir):
    """seed_<n>[_<i>] subdirectories of exp_dir grouped by rerun index i (0
    when un-numbered): {i: [(n, dir), ...]}, by ascending i and n."""
    groups = {}
    for d in exp_dir.iterdir():
        if d.is_dir() and (m := re.fullmatch(r"seed_(\d+)(?:_(\d+))?", d.name)):
            groups.setdefault(int(m.group(2) or 0), []).append((int(m.group(1)), d))
    return {i: sorted(g) for i, g in sorted(groups.items())}


def npz_name(rerun):
    return "seeds.npz" if rerun == 0 else f"seeds_{rerun:02d}.npz"


def load_seed(seed, run_dir):
    """Everything one seed run saved, keyed as in seeds.npz, or None while it
    is incomplete."""
    ck = run_dir / "checkpoints"
    split_path = ck / "mesh_split.json"
    if not split_path.exists():
        return None
    test_meshes = json.loads(split_path.read_text())["test"]
    names = ["model.pt", "train_losses.npy", "val_losses.npy", "lrs.npy",
             "msg_reg_log.npz"] + [f"rollout_mae_{m}.npy" for m in test_meshes]
    if not all((ck / n).exists() for n in names):
        return None

    state = torch.load(ck / "model.pt", map_location="cpu", weights_only=True)
    row = {
        "seeds": seed,
        "test_meshes": np.array(test_meshes),
        # delta_scale is a buffer, not a parameter
        "n_params": sum(v.numel() for k, v in state.items()
                        if not k.endswith("delta_scale")),
        "train_losses": np.load(ck / "train_losses.npy"),
        "val_losses": np.load(ck / "val_losses.npy"),
        "lrs": np.load(ck / "lrs.npy"),
    }
    with np.load(ck / "msg_reg_log.npz") as log:
        for k in log.files:
            row[f"msg_{k}"] = log[k]
    for m in test_meshes:
        row[f"rollout_mae_{m}"] = np.load(ck / f"rollout_mae_{m}.npy")
    return row


def stack(rows):
    """One array per key: stacked along the seed axis, or a single copy for
    what every seed must share."""
    per_epoch = {f"msg_{k}" for k in MessageRegularizer.LOG_KEYS}
    per_seed = {f"msg_{k}" for k in MSG_PER_SEED}
    shared = {k for k in rows[0]
              if k.startswith("msg_") and k not in per_epoch | per_seed}
    shared |= {"test_meshes", "n_params"}

    keys = set(rows[0])
    for r in rows[1:]:
        if set(r) != keys:
            raise Inconsistent(f"seed {r['seeds']} saved {sorted(set(r) ^ keys)}"
                               f" differently from seed {rows[0]['seeds']}")
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if k in shared:
            differ = [r["seeds"] for r in rows if not np.array_equal(r[k], vals[0])]
            if differ:
                raise Inconsistent(f"{k} of seeds {differ} differs from seed "
                                   f"{rows[0]['seeds']}'s ({vals[0]})")
            out[k] = np.asarray(vals[0])
        else:
            shapes = {np.shape(v) for v in vals}
            if len(shapes) > 1:
                raise Inconsistent(f"{k} has shapes {sorted(shapes)} across seeds")
            out[k] = np.stack(vals)
    return out


def aggregate(exp_dir, reruns=None):
    """Write exp_dir/seeds.npz, seeds_01.npz, ..., one per rerun index, or only
    those in reruns. Returns False if the seeds of any are inconsistent."""
    groups = seed_dirs(exp_dir)
    ok = True
    for rerun in (sorted(groups) if reruns is None else reruns):
        name = npz_name(rerun)
        rows, unfinished = [], []
        for seed, d in groups.get(rerun, []):
            row = load_seed(seed, d)
            if row is None:
                unfinished.append(d.name)
            else:
                rows.append(row)
        note = f" (unfinished, left out: {', '.join(unfinished)})" if unfinished else ""
        if not rows:
            print(f"{exp_dir.name}: no finished seed for {name} yet{note}")
            continue
        try:
            out = stack(rows)
        except Inconsistent as e:
            print(f"{exp_dir.name}: NOT aggregated, {name} left as it was: {e}")
            ok = False
            continue
        out["exp_name"] = np.array(exp_dir.name.removeprefix(f"{case_name}_"))
        np.savez(exp_dir / name, **out)
        print(f"{exp_dir.name}: seeds {out['seeds'].tolist()} -> {name}{note}")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # <exp_name>[:<i>]; the same experiment may come with several indices
        targets = {}
        for arg in sys.argv[1:]:
            name, _, rerun = arg.partition(":")
            d = processed_data_dir / f"{case_name}_{name.removeprefix(case_name + '_')}"
            if not rerun:
                targets[d] = None
            elif targets.get(d, []) is not None:
                targets.setdefault(d, []).append(int(rerun))
        missing = [str(d) for d in targets if not d.is_dir()]
        if missing:
            sys.exit(f"No such experiment directory: {', '.join(missing)}")
        targets = {d: r and sorted(set(r)) for d, r in targets.items()}
    else:
        targets = {d: None for d in sorted(processed_data_dir.glob(f"{case_name}_*"))
                   if d.is_dir() and seed_dirs(d)}
        if not targets:
            sys.exit(f"No experiment in {processed_data_dir} has seed_<n> runs yet.")
    ok = [aggregate(d, r) for d, r in targets.items()]
    sys.exit(0 if all(ok) else 1)
