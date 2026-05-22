import torch
import numpy as np
from mesh2graph.utils import parse_internal_fields_alltimes, parse_boundary_fields_alltimes, parse_vertex_centered, add_boundary_points, filter_of_time_directories

def load_fields(case_dir, field_name, excluded_patches) -> torch.Tensor:
    static_graph = parse_vertex_centered(case_dir)
    num_internal_nodes = static_graph.num_nodes
    static_graph, _, patches = add_boundary_points(static_graph, case_dir, excluded_patches, return_face_idx_patch=True)
    num_timesteps = len(filter_of_time_directories(case_dir))

    Field = torch.zeros(num_timesteps, static_graph.num_nodes) # (T, N)

    field_int   = parse_internal_fields_alltimes(case_dir, field_name)[field_name]                                    #dict
    field_bound = parse_boundary_fields_alltimes(case_dir, field_name, excluded_patches=excluded_patches)[field_name] #dict



    for i, t in enumerate(field_bound.keys()):
        Field_t = torch.zeros(static_graph.num_nodes)
        if isinstance(field_int[t], (int, float)):
            field_int[t] = np.full(num_internal_nodes, field_int[t])
        Field_t[:num_internal_nodes] = torch.from_numpy(field_int[t]) # internal nodes

        for patch_name in patches:
            patch_idx = patches[patch_name]
            Field_t[patch_idx] = torch.from_numpy(field_bound[t][bytes(patch_name, "utf-8")]).float() # boundary nodes

        Field[i, :] = Field_t

    return Field