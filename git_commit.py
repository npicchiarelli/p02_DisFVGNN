import json
import os
import subprocess
from pathlib import Path


# Saved with each checkpoint: the commit the run was trained with.
GIT_COMMIT = "git_commit.json"

REPO_DIR = Path(__file__).resolve().parent


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True,
                          text=True, check=True).stdout.strip()


def save_git_commit(checkpoint_dir):
    # Call it at the start of training: that is the code the run imported.
    # "dirty" flags uncommitted changes to tracked files, in which case the
    # commit alone does not reproduce the run. Untracked files are ignored.
    try:
        record = {"commit": _git("rev-parse", "HEAD"),
                  "dirty": bool(_git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError) as err:
        print(f"WARNING: could not read the git commit ({err}); recording none")
        record = {"commit": None, "dirty": None}
    with open(os.path.join(checkpoint_dir, GIT_COMMIT), "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"Commit: {record['commit']}" + (" (uncommitted changes)" if record["dirty"] else ""))


def load_git_commit(checkpoint_dir) -> dict | None:
    # None: a checkpoint older than git_commit.json.
    path = os.path.join(checkpoint_dir, GIT_COMMIT)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)
