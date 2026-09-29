# CLAUDE.md

## `data_analysis.ipynb`

- Every markdown comment that describes a training run must include the short hash (first 7
  characters) of the commit that run was trained with, next to the run name, e.g.
  `parametric_history1_msg_dim64_residual` (commit `2f3080c`). If a comment compares several
  runs, give the hash for each one.
- Read the hash from the run's checkpoints: `train.py` and `train_parametric.py` write
  `checkpoints/git_commit.json` (`{"commit": ..., "dirty": ...}`) at the start of training,
  under `../processed_data/<run>/checkpoints/` or `../processed_data/<run>/seed_<n>/checkpoints/`.
  `git_commit.load_git_commit(checkpoint_dir)` reads it.
- If `"dirty"` is `true`, the run had uncommitted changes, so write that next to the hash
  (commit `2f3080c` + uncommitted changes).
- Checkpoints trained before `git_commit.json` existed do not have it, and neither do
  checkpoints that are not stored locally. Never guess the hash from the current `HEAD` or the
  run's date: ask for it.
