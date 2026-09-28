import hashlib
import os
import sys
from pathlib import Path

import torch

import mesh2graph.utils
from . import field, static_graph
from .field import load_fields
from .static_graph import build_static_graph


def code_key() -> str:
    """Short hash of the code that builds graphs and fields, and of this module.
    Editing any of it invalidates the cache. Files are located through the
    imported modules, so a copied script still finds them."""
    files = [mesh2graph.utils.__file__, sys.modules["getter_of"].__file__,
             static_graph.__file__, field.__file__, __file__]
    return hashlib.sha1(b"".join(Path(f).read_bytes() for f in files)).hexdigest()[:8]


def load_mesh_cached(case_dir, excluded_patches, cache_dir, use_cache=True):
    """Return (static_graph, T_sequence) for one case, cached in cache_dir.

    Keyed by case, excluded patches and code_key(). Raw-data changes are not
    detected: clear cache_dir or pass use_cache=False. Graphs keep all edge
    features; callers drop the FV ones if needed."""
    if not use_cache:
        return (build_static_graph(case_dir, excluded_patches),
                load_fields(case_dir, "T", excluded_patches=excluded_patches))

    name = os.path.basename(os.path.normpath(case_dir))
    excl_key = "-".join(excluded_patches) if excluded_patches else "none"
    cache_file = Path(cache_dir) / f"{name}__T__{excl_key}__{code_key()}.pt"
    if cache_file.exists():
        print(f"  from cache {cache_file.name}")
        blob = torch.load(cache_file, weights_only=False)
        return blob["graph"], blob["T"]

    g = build_static_graph(case_dir, excluded_patches)
    T = load_fields(case_dir, "T", excluded_patches=excluded_patches)
    # Write, then rename: concurrent runs never read a partial file.
    os.makedirs(cache_dir, exist_ok=True)
    tmp_file = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
    torch.save({"graph": g, "T": T}, tmp_file)
    os.replace(tmp_file, cache_file)
    return g, T
