import json
import os

import torch
import numpy as np
from torch_geometric.data import Data

from mesh2graph.utils import parse_vertex_centered, add_boundary_points, load_face_surfaces_by_patch, load_face_owners, load_skewness, load_non_orthogonality, load_cell_volumes
from smithers.io.openfoam import FoamMesh
from smithers.io.openfoam import field_parser


# Saved with each checkpoint: how its graphs were built.
GRAPH_CONFIG = "graph_config.json"


def save_graph_config(checkpoint_dir, boundary_pos, excluded_patches, cell_volume):
    with open(os.path.join(checkpoint_dir, GRAPH_CONFIG), "w") as fh:
        json.dump({"boundary_pos": boundary_pos,
                   "excluded_patches": list(excluded_patches),
                   "cell_volume": cell_volume}, fh, indent=2)


def load_graph_config(checkpoint_dir) -> dict:
    # No file: a checkpoint older than graph_config.json, hence point_mean.
    # No cell_volume key: a checkpoint older than the cell-volume node feature.
    path = os.path.join(checkpoint_dir, GRAPH_CONFIG)
    if not os.path.exists(path):
        return {"boundary_pos": "point_mean", "cell_volume": False}
    with open(path) as fh:
        return {"cell_volume": False, **json.load(fh)}


def node_feature_columns(cell_volume: bool, positions: bool) -> list:
    # node_attr is [node type (2), log cell volume (1), x, y, z (3)]; the columns a
    # model takes. The cell volume is an FV feature, dropped with the FV edge
    # features; checkpoints older than it take cell_volume=False.
    return [0, 1] + ([2] if cell_volume else []) + ([3, 4, 5] if positions else [])


def build_static_graph(case_dir,
                       excluded_patches,
                       boundary_edge_dir: str = "both",
                       boundary_pos: str = "cf") -> Data:
    # boundary_edge_dir orients the cell <-> boundary-node edges, named from the
    # boundary node's point of view: "source" (BC propagates inward only),
    # "sink" (boundary node only receives, so the BC cannot affect the
    # prediction), or "both" (default). See add_boundary_points.
    # boundary_pos: "cf" (default) or "point_mean", see add_boundary_points.

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
    log_vol = torch.log(load_cell_volumes(case_dir))
    cell_vol = torch.cat((log_vol, log_vol.mean().expand(N_bnd, 1)), dim = 0)

    # Adding node type, log cell volume and position to the x attribute of static_graph.
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

    return static_graph
