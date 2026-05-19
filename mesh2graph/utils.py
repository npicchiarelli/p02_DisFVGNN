import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode

import torch
import torch_geometric
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
from smithers.io.openfoam import FoamMesh
from getter_of import getter_of


def parse_vertex_centered(directory: str, return_face_idx: bool = False) -> Data:

    mesh = FoamMesh(directory)
    try:
        mesh.read_cell_centres(os.path.join(directory, '0/C'))
    except:
        print("Error reading cell centres\nPlease execute `postProcess -func 'writeCellCentres' -time 0' in the mesh directory to generate the file `0/C` containing the cell centres")
    u, v = [], []
    if return_face_idx:
        face_idx = []
    for src in range(mesh.num_cell):
        for dst in mesh.cell_neighbour_cells(src):
            # skip boundary neighbors
            if dst < 0:
                continue
            u.append(src)
            v.append(dst)
            face_idx.append(set(mesh.cell_faces[src]).intersection(set(mesh.cell_faces[dst])).pop()) # get the index of the face shared by the two cells

    u = torch.tensor(u)
    v = torch.tensor(v)

    graph_vc = Data(x = torch.arange(mesh.num_cell).reshape(-1, 1), pos = torch.tensor(mesh.cell_centres[:,:]), edge_index=torch.stack([u, v], dim=0))
    if return_face_idx:
        return graph_vc, face_idx
    else:
        return graph_vc

def parse_cell_centered(directory: str) -> Data:
    mesh = FoamMesh(directory)
    u, v = [], []
    for src in range(mesh.num_point):
        incident_faces = [(i,face) for (i,face) in enumerate(mesh.faces) if src in face] # list of tuples (I may need the face index afterwards)
        incident_faces_points = set()

        for i,face in incident_faces:
            idx = face.index(src)
            incident_faces_points.add(face[idx-1])
            incident_faces_points.add(face[(idx+1) % len(face)])
        for dst in sorted(incident_faces_points):
            u.append(src)
            v.append(dst)

    u = torch.tensor(u)
    v = torch.tensor(v)

    graph_cc = Data(x = torch.arange(mesh.num_point).reshape(-1, 1), pos = torch.tensor(mesh.points[:,:]), edge_index=torch.stack([u, v], dim=0))

    return graph_cc

def load_boundary_face_centers(directory: str, verbose: bool = False) -> dict:

    face_centres = dict()

    of_binder = getter_of([".", "-case", f"{directory}"])

    Cf = of_binder.getCf()
    names = of_binder.getPatchName()

    for i,CfPatch in enumerate(Cf):        
        patchName = names[i]
        
        Cfx = CfPatch[0::3, 0]
        Cfy = CfPatch[1::3, 0]
        Cfz = CfPatch[2::3, 0]

        face_centres[patchName] = torch.from_numpy(np.stack([Cfx, Cfy, Cfz], axis=1))

    if verbose:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print(f"Mesh folder: {dir_path}")

    return face_centres

def load_face_surfaces(graph_vc: Data, directory: str) -> torch.Tensor:

    of_binder = getter_of([".", "-case", f"{directory}"])

    Sf = of_binder.getSf()
    Sf = torch.from_numpy(Sf.reshape(-1, 3))

    return Sf

def load_face_surfaces_by_patch(graph_vc: Data, directory: str) -> dict:

    of_binder = getter_of([".", "-case", f"{directory}"])

    Sf = of_binder.getSfByPatch()
    Sf = [SfPatch.reshape(-1, 3) for SfPatch in Sf]
    names = of_binder.getPatchName()
    names.insert(0, 'internal')

    return dict(zip(names, Sf))

def load_skewness(directory: str) -> torch.Tensor:

    of_binder = getter_of([".", "-case", f"{directory}"])

    skewness = of_binder.getSkewness()
    skewness = torch.from_numpy(skewness.reshape(-1, 1))

    return skewness

def load_non_orthogonality(directory: str) -> torch.Tensor:

    of_binder = getter_of([".", "-case", f"{directory}"])

    non_orthogonality = of_binder.getNonOrthogonality()
    non_orthogonality = torch.from_numpy(non_orthogonality.reshape(-1, 1))

    return non_orthogonality

def add_boundary_points(graph_vc: Data, directory: str, excluded_faces: list, return_face_idx: bool = False, verbose: bool = False) -> Data:

    of_binder = getter_of([".", "-case", f"{directory}"])
    names = of_binder.getPatchName()
    
    mesh = FoamMesh(directory)
    u,v = [], []
    if return_face_idx:
        boundary_faces_idx = []
    boundary_index = mesh.num_cell-1 # boundary nodes will have indices out of the bounds of the cell-centered graph
    pos_dict = {}
    for i, cell in enumerate(mesh.cell_faces):
        for boundary_name in names:
            if boundary_name in excluded_faces:
                continue
            if mesh.is_cell_on_boundary(i, bytes(boundary_name, "utf-8")):
                for face in mesh.cell_faces[i]:
                    if mesh.is_face_on_boundary(face,  bytes(boundary_name, "utf-8")):
                        if verbose:
                            print(f"Cell {i} is on the {boundary_name} boundary and has face {face} on the {boundary_name} boundary")
                        boundary_index += 1
                        pos_dict[boundary_index] = np.mean(mesh.points[mesh.faces[face]], axis=0)
                        if return_face_idx:
                            boundary_faces_idx.append(face)
                        u.append(i)
                        v.append(boundary_index)
    
    coo_index_b = torch.stack([torch.tensor(u), torch.tensor(v)], dim=0)
    x_b = torch.cat([graph_vc.x, torch.from_numpy(np.array(v)).unsqueeze(1)])

    pos_b = torch.cat([graph_vc.pos, torch.from_numpy(np.array([pos_dict[i] for i in range(mesh.num_cell, boundary_index+1)]))], dim=0)

    edge_index_b = torch.cat([graph_vc.edge_index, coo_index_b], dim=1)

    graph_vc_boundary = Data(x = x_b, pos = pos_b, edge_index=edge_index_b)

    if return_face_idx:
        return graph_vc_boundary, boundary_faces_idx
    else:
        return graph_vc_boundary

def make_undirected(coo:torch.Tensor):
    if coo.ndim != 2 or coo.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], "
            f"got {tuple(coo.shape)}"
        )
    src, dst = coo
    undirected = torch.stack([
        torch.minimum(src, dst),
        torch.maximum(src, dst)
    ])
    return torch.unique(undirected, dim = 1)
