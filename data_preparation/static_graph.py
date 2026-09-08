import torch
import numpy as np
from torch_geometric.data import Data

from mesh2graph.utils import parse_vertex_centered, add_boundary_points, load_face_surfaces_by_patch, load_face_owners, load_skewness, load_non_orthogonality
from smithers.io.openfoam import FoamMesh
from smithers.io.openfoam import field_parser



def build_static_graph(case_dir,
                       excluded_patches,
                       boundary_edge_dir: str = "both") -> Data:
    # boundary_edge_dir orients the cell <-> boundary-node edges, named from the
    # boundary node's point of view: "source" (BC propagates inward only),
    # "sink" (boundary node only receives, so the BC cannot affect the
    # prediction), or "both" (default). See add_boundary_points.

    #-------------------------------------NODES---------------------------------------------

    #At this point our graph only has the structure: number of internal nodes and connectivity
    static_graph, innner_faces_idx = parse_vertex_centered(case_dir, return_face_idx=True)
    N_int = static_graph.num_nodes # Number of internal nodes

    
    node_type = torch.zeros(N_int, 2, dtype=torch.float32)
    node_type[:,0] = 1.0 # One hot encoding for internal or boundary type. Only internal nodes at this point

    static_graph, boundary_faces_idx, patches = add_boundary_points(static_graph, case_dir, excluded_patches, return_face_idx_patch=True, boundary_edge_dir=boundary_edge_dir)
    N_bnd = static_graph.num_nodes - N_int

    bnd_type = torch.zeros(N_bnd, 2, dtype=torch.float32)
    bnd_type[:, 1] = 1.0 # One hot encoding for boundary nodes
    node_type = torch.cat((node_type, bnd_type), dim = 0) # add them to the node_type encoding

    # Adding node type and position to the x attribute of static_graph
    static_graph.node_attr = torch.cat((node_type, static_graph.pos), dim = 1).float()

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

    # getNonOrthogonality in getter_of.C hardcodes every boundary face to 1.0
    # ("perfectly orthogonal") because a boundary face has no neighbour cell.
    # That asserts perfect orthogonality exactly where the geometry is worst: on
    # the parametric meshes the true value reaches cos ~ 0.17 (80 deg), so the
    # feature is a dead constant on precisely the edges that carry the boundary
    # condition. Recompute it with the boundary NODE (the face centre) standing
    # in for the missing neighbour centre — the same definition OpenFOAM uses
    # internally, cos(Sf, C_neighbour - C_owner). Sf and dist_vec are both
    # oriented source -> target, so the ratio is invariant under edge direction
    # and matches the internal-face values to ~1e-5 (0/C is ascii, writePrecision 6).
    is_boundary_edge = (src >= N_int) | (dst >= N_int)
    cos_ortho = (surface_area_vec.double() * dist_vec.double()).sum(dim=1, keepdim=True) / (
        surface_area_vec_norm.double() * dist_norm.double())
    No[is_boundary_edge] = cos_ortho.clamp(-1.0, 1.0)[is_boundary_edge].to(No.dtype)
    edge_attr = torch.cat((dist_vec, dist_norm, surface_area_vec, surface_area_vec_norm, Sk, No), dim=1)
    static_graph.edge_attr = edge_attr.float()

    return static_graph
