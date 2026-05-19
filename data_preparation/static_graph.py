import torch
import numpy as np
from torch_geometric.data import Data

from mesh2graph.utils import parse_vertex_centered, add_boundary_points, load_face_surfaces_by_patch, load_skewness, load_non_orthogonality
from smithers.io.openfoam import FoamMesh
from smithers.io.openfoam import field_parser



def build_static_graph(case_dir,
                       excluded_faces) -> Data:
    
    #-------------------------------------NODES---------------------------------------------

    #At this point our graph only has the structure: number of internal nodes and connectivity
    static_graph, innner_faces_idx = parse_vertex_centered(case_dir, return_face_idx=True)
    N_int = static_graph.num_nodes # Number of internal nodes

    
    node_type = torch.zeros(N_int, 2, dtype=torch.float32)
    node_type[:,0] = 1.0 # One hot encoding for internal or boundary type. Only internal nodes at this point

    static_graph, boundary_faces_idx = add_boundary_points(static_graph, case_dir, excluded_faces, return_face_idx=True)
    N_bnd = static_graph.num_nodes - N_int

    bnd_type = torch.zeros(N_bnd, 2, dtype=torch.float32)
    bnd_type[:, 1] = 1.0 # One hot encoding for boundary nodes
    node_type = torch.cat((node_type, bnd_type), dim = 0) # add them to the node_type encoding

    # #Adding node type and position to the x attribute of static_graph
    static_graph.node_attr = torch.cat((node_type, static_graph.pos), dim = 1)

    #-------------------------------------EDGES---------------------------------------------

    # Distance calculation
    owner    = static_graph.edge_index[:,0]
    neighbor = static_graph.edge_index[:,1]

    dist_vec  = static_graph.pos[owner] - neighbor[neighbor]
    dist_norm = np.linalg.norm(dist_vec, axis=1, keepdims=True)

    # Surface area vector from openfoam

    Sf_by_patch = load_face_surfaces_by_patch(static_graph, case_dir)
    Sf = np.concatenate([Sf_by_patch[s] if s not in excluded_faces 
                            else np.full(Sf_by_patch[s].shape, np.nan) 
                            for s in Sf_by_patch], axis=0)
    surface_area_vec = Sf[innner_faces_idx+boundary_faces_idx]
    surface_area_vec = torch.from_numpy(surface_area_vec).float()
    surface_area_vec_norm = torch.norm(surface_area_vec, dim=1, keepdim=True)

    Sk = load_skewness(case_dir)
    No = load_non_orthogonality(case_dir)
    Sk = Sk[innner_faces_idx+boundary_faces_idx]
    No = No[innner_faces_idx+boundary_faces_idx]
    edge_attr = torch.cat((dist_vec, dist_norm, surface_area_vec, surface_area_vec_norm, Sk, No), dim=1)

    return Data(x=static_graph.node_attr, edge_index=static_graph.edge_index, edge_attr=edge_attr)
