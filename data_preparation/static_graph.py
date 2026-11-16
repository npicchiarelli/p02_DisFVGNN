import json
import os

import torch
import numpy as np
from torch_geometric.data import Data

from mesh2graph.utils import parse_vertex_centered, add_boundary_points, load_face_surfaces_by_patch, load_face_owners, load_skewness, load_non_orthogonality, load_cell_volumes, load_laplacian, load_boundary_value_coeffs
from mesh2graph.laplacian_eigen import laplacian_eigenpairs, boundary_face_values
from smithers.io.openfoam import FoamMesh
from smithers.io.openfoam import field_parser


# Saved with each checkpoint: how its graphs were built.
GRAPH_CONFIG = "graph_config.json"

# How the cell volume V becomes a node feature, see build_static_graph.
CELL_VOLUME_TRANSFORMS = ("log", "cbrt")

# Laplacian eigenmodes (mesh2graph/laplacian_eigen.py). The operator is
# -laplacian(EIG_COEFF, EIG_FIELD) with EIG_FIELD's own BCs, read at EIG_TIME.
# The cached graphs hold EIG_K_MAX modes; a run takes the first n_eig of them.
EIG_K_MAX = 32
EIG_FIELD = "T"
EIG_COEFF = "DT"
EIG_TIME  = "0"


def save_graph_config(checkpoint_dir, boundary_pos, excluded_patches, cell_volume, cell_volume_transform="log", dataset="parametric", n_eig=0):
    with open(os.path.join(checkpoint_dir, GRAPH_CONFIG), "w") as fh:
        json.dump({"boundary_pos": boundary_pos,
                   "excluded_patches": list(excluded_patches),
                   "cell_volume": cell_volume,
                   "n_eig": n_eig,
                   "eig_field": EIG_FIELD,
                   "eig_coeff": EIG_COEFF,
                   "eig_time": EIG_TIME,
                   "cell_volume_transform": cell_volume_transform,
                   "dataset": dataset}, fh, indent=2)


def load_graph_config(checkpoint_dir) -> dict:
    # No file: a checkpoint older than graph_config.json, hence point_mean.
    # No cell_volume key: a checkpoint older than the cell-volume node feature.
    # No n_eig key: a checkpoint older than the eigenmode node features.
    # No cell_volume_transform key: older than the option, hence log.
    # No dataset key: older than the option, so raw_data/parametric as it was at
    # training time. Runs before 2026-11-09 saw the dt = 0.025 data now in
    # raw_data/parametric_old, not the dt = 0.005 data now in raw_data/parametric.
    defaults = {"cell_volume": False, "cell_volume_transform": "log", "dataset": "parametric", "n_eig": 0}
    path = os.path.join(checkpoint_dir, GRAPH_CONFIG)
    if not os.path.exists(path):
        return {"boundary_pos": "point_mean", **defaults}
    with open(path) as fh:
        return {**defaults, **json.load(fh)}


def node_feature_columns(cell_volume: bool, positions: bool) -> list:
    # node_attr is [node type (2), cell volume feature (1), x, y, z (3)]; the columns a
    # model takes. The cell volume is an FV feature, dropped with the FV edge
    # features; checkpoints older than it take cell_volume=False.
    return [0, 1] + ([2] if cell_volume else []) + ([3, 4, 5] if positions else [])


def node_features(g: Data, cell_volume: bool, positions: bool, n_eig: int = 0) -> torch.Tensor:
    """The node features a model takes: the node_feature_columns of node_attr,
    then the first n_eig Laplacian eigenmodes, always LAST (eig_columns relies on
    it). The modes live in g.eig_vecs rather than node_attr, so scripts that feed
    the whole node_attr to a model (train.py, test_flange.py) are unaffected."""
    x = g.node_attr[:, node_feature_columns(cell_volume, positions)]
    if n_eig == 0:
        return x
    if not hasattr(g, "eig_vecs") or g.eig_vecs.shape[1] < n_eig:
        have = g.eig_vecs.shape[1] if hasattr(g, "eig_vecs") else 0
        raise ValueError(f"n_eig = {n_eig} but the graph holds {have} eigenmodes "
                         f"(built with eig_modes = {have}, at most EIG_K_MAX = {EIG_K_MAX})")
    return torch.cat((x, g.eig_vecs[:, :n_eig]), dim=1)


def eig_columns(n_features: int, n_eig: int) -> list:
    """Indices of the eigenmode columns in node_features' output."""
    return list(range(n_features - n_eig, n_features))


def build_static_graph(case_dir,
                       excluded_patches,
                       boundary_edge_dir: str = "both",
                       boundary_pos: str = "cf",
                       eig_modes: int = 0,
                       cell_volume_transform: str = "log") -> Data:
    # boundary_edge_dir orients the cell <-> boundary-node edges, named from the
    # boundary node's point of view: "source" (BC propagates inward only),
    # "sink" (boundary node only receives, so the BC cannot affect the
    # prediction), or "both" (default). See add_boundary_points.
    # boundary_pos: "cf" (default) or "point_mean", see add_boundary_points.
    # eig_modes > 0 also stores that many Laplacian eigenmodes in g.eig_vecs
    # (N, eig_modes) and their eigenvalues [1/s] in g.eig_vals; see add_eigenmodes.
    # cell_volume_transform: "log" (default) or "cbrt", see the cell volume below.
    if cell_volume_transform not in CELL_VOLUME_TRANSFORMS:
        raise ValueError(f"cell_volume_transform must be one of {CELL_VOLUME_TRANSFORMS}, "
                         f"got {cell_volume_transform!r}")

    #-------------------------------------NODES---------------------------------------------

    #At this point our graph only has the structure: number of internal nodes and connectivity
    static_graph, innner_faces_idx = parse_vertex_centered(case_dir, return_face_idx=True)
    N_int = static_graph.num_nodes # Number of internal nodes

    
    node_type = torch.zeros(N_int, 2, dtype=torch.float32)
    node_type[:,0] = 1.0 # One hot encoding for internal or boundary type. Only internal nodes at this point

    static_graph, boundary_faces_idx, patches = add_boundary_points(static_graph, case_dir, excluded_patches, return_face_idx_patch=True, boundary_edge_dir=boundary_edge_dir, boundary_pos=boundary_pos)
    N_bnd = static_graph.num_nodes - N_int

    bnd_type = torch.zeros(N_bnd, 2, dtype=torch.float32)
    bnd_type[:, 1] = 1.0 # One hot encoding for boundary nodes
    node_type = torch.cat((node_type, bnd_type), dim = 0) # add them to the node_type encoding

    # Cell volume: internal nodes are the cells in OpenFOAM cell order. Log, as the
    # snappyHexMesh refinement levels are 8x apart (~3500x overall on the parametric
    # meshes), which a z-score alone would squash. Boundary nodes are face centres
    # and have no volume: node_type tells them apart, so they take the mean internal
    # log volume, which keeps them out of the way of the normalizer's statistics
    # (a 0 in log space would sit ~20 away and dominate the std).
    # "cbrt" instead takes V^(1/3), the cell length h, on a linear scale, and 0 on
    # the boundary nodes: a face centre is a cell of zero thickness. 0 sits inside
    # the internal range once z-scored (~-1.5 on model_000, internal -1.1..+4.8),
    # but the fine levels end up closer together than in log (levels 0/1: 0.2
    # apart against 0.6) and the coarse far field becomes the tail.
    # The z-scored log h is the same feature as the z-scored log V.
    vol = load_cell_volumes(case_dir)
    if cell_volume_transform == "log":
        vol_int = torch.log(vol)
        vol_bnd = vol_int.mean().expand(N_bnd, 1)
    else:
        vol_int = vol ** (1.0 / 3.0)
        vol_bnd = torch.zeros(N_bnd, 1, dtype=vol_int.dtype)
    cell_vol = torch.cat((vol_int, vol_bnd), dim = 0)

    # Adding node type, cell volume feature and position to the x attribute of static_graph.
    # The column order is relied on by node_feature_columns.
    static_graph.node_attr = torch.cat((node_type, cell_vol.float(), static_graph.pos.float()), dim = 1).float()

    #-------------------------------------EDGES---------------------------------------------

    # Distance calculation. src/dst are the edge endpoints under PyG's default
    # source_to_target flow (messages are aggregated at dst); they are NOT the
    # OpenFOAM face owner/neighbour, which is a per-face property used below.
    src = static_graph.edge_index[0,:]
    dst = static_graph.edge_index[1,:]

    # Oriented source -> target, the same direction as Sf below. This matches
    # OpenFOAM's own pairing d = C_neighbour - C_owner with Sf pointing owner ->
    # neighbour, so Sf . dist_vec > 0 and the non-orthogonality of a face is the
    # angle between the two.
    dist_vec  = static_graph.pos[dst] - static_graph.pos[src]
    dist_norm = dist_vec.norm(dim=1, keepdim=True)

    # Surface area vector from openfoam
    Sf_by_patch = load_face_surfaces_by_patch(static_graph, case_dir)
    Sf = np.concatenate([Sf_by_patch[s] if s not in excluded_patches
                            else np.full(Sf_by_patch[s].shape, np.nan)
                            for s in Sf_by_patch], axis=0) # useful for debugging, but not strictly necessary since we will be selecting only the faces that are not in the excluded_faces list

    # A face's points are stored counter-clockwise as seen from inside its owner
    # cell, so OpenFOAM's Sf points out of the owner and into the neighbour (out
    # of the domain for a boundary face). Every face becomes two opposite
    # directed edges here, so Sf must be negated on the direction that does not
    # start at the owner. Convention: Sf points from the edge SOURCE to the edge
    # TARGET, i.e. it is the INWARD area vector of the receiving node — the
    # orientation an FV flux into that cell is computed with, so that summing
    # messages at a node mirrors the FV flux balance over its faces.
    face_of_edge  = np.asarray(innner_faces_idx + boundary_faces_idx)
    owner_of_edge = load_face_owners(case_dir)[face_of_edge]
    orientation   = np.where(src.numpy() == owner_of_edge, 1.0, -1.0)[:, None]

    surface_area_vec = torch.from_numpy(Sf[face_of_edge] * orientation).float()
    surface_area_vec_norm = torch.norm(surface_area_vec, dim=1, keepdim=True)

    Sk = load_skewness(case_dir)
    No = load_non_orthogonality(case_dir)
    Sk = Sk[innner_faces_idx+boundary_faces_idx]
    No = No[innner_faces_idx+boundary_faces_idx]

    # getNonOrthogonality (getter_of.C), like OpenFOAM, sets boundary faces to
    # 1.0: they have no neighbour cell. That hides the worst angles (cos ~ 0.17
    # on the parametric meshes), so recompute cos(Sf, d) with the boundary node
    # standing in for the neighbour centre, as for internal faces. Sf and
    # dist_vec are both oriented source -> target, so the ratio does not depend
    # on edge direction. With "cf", |d| * cos equals OpenFOAM's 1/deltaCoeffs.
    is_boundary_edge = (src >= N_int) | (dst >= N_int)
    cos_ortho = (surface_area_vec.double() * dist_vec.double()).sum(dim=1, keepdim=True) / (
        surface_area_vec_norm.double() * dist_norm.double())
    No[is_boundary_edge] = cos_ortho.clamp(-1.0, 1.0)[is_boundary_edge].to(No.dtype)
    edge_attr = torch.cat((dist_vec, dist_norm, surface_area_vec, surface_area_vec_norm, Sk, No), dim=1)
    static_graph.edge_attr = edge_attr.float()

    if eig_modes > 0:
        add_eigenmodes(static_graph, case_dir, boundary_faces_idx[:N_bnd],
                       excluded_patches, eig_modes)

    return static_graph


def add_eigenmodes(static_graph: Data, case_dir, bnd_faces, excluded_patches, k: int):
    """Eigenmodes of K phi = lam diag(V) phi, K = -laplacian(EIG_COEFF, EIG_FIELD)
    with the field's own BCs (getLaplacian in getter_of.C).

    Internal nodes are the cells, in cell order. Boundary node j is face
    bnd_faces[j] (add_boundary_points creates one node per face, in that order),
    and takes the face value of the homogeneous BC, phi_f = c_f * phi_owner:
    0 on a fixedValue patch, the owner value on a zeroGradient one.
    Modes are V-orthonormal, phi.T @ (V * phi) = I, and sign-canonical."""
    K = load_laplacian(case_dir, EIG_FIELD, EIG_COEFF, EIG_TIME)
    V = load_cell_volumes(case_dir).numpy().ravel()
    lam, phi = laplacian_eigenpairs(K, V, k)

    coeffs = load_boundary_value_coeffs(case_dir, EIG_FIELD, EIG_TIME)
    # Boundary nodes on a patch whose BC does not fix the value are fine for the
    # modes (they take the owner value), but the graph treats every boundary
    # node as a BC carrier, so say so.
    for name, c in coeffs.items():
        if name not in excluded_patches and len(c) and not np.all(c == 0):
            print(f"  note: patch {name!r} has graph boundary nodes but its {EIG_FIELD} "
                  f"BC does not fix the value (c in [{np.nanmin(c):g}, {np.nanmax(c):g}])")
    phi_b = boundary_face_values(phi, bnd_faces, load_face_owners(case_dir), list(coeffs.values()))

    static_graph.eig_vecs = torch.from_numpy(np.concatenate((phi, phi_b), axis=0)).float()
    static_graph.eig_vals = torch.from_numpy(lam).float()
    assert static_graph.eig_vecs.shape[0] == static_graph.num_nodes
