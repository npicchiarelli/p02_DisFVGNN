import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import torch_geometric

def plot_graph(graph, savepath:str=None):
    graph_nx = torch_geometric.utils.to_networkx(graph, to_undirected=False)
    draw_pos = {i: graph.pos[i].numpy() for i in range(graph.num_nodes)}

    px = 1/plt.rcParams['figure.dpi']  # convert pixels to inches
    fig, ax = plt.subplots(figsize=(1920*px, 1080*px))
    ax.set_aspect('equal')
    nx.draw(graph_nx, pos = draw_pos, ax = ax, node_size=20)

    ax.set_axis_on()                   # ensure axes are visible
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)

def plot_graphs_3d(graphs, titles=None, savepath: str = None):

    """
    graphs : list of PyTorch Geometric Data objects
    titles : optional list of legend labels, one per graph
    """
    n = len(graphs)
    if titles is None:
        titles = [f"Graph {i+1}" for i in range(n)]

    colors = ['#7a4444', '#0d2448', 'mediumpurple', 'orange']

    fig = go.Figure()

    for i, graph in enumerate(graphs):
        color = colors[i % len(colors)]
        graph_nx = torch_geometric.utils.to_networkx(graph, to_undirected=False)
        pos_3d = {j: graph.pos[j].numpy() for j in range(graph.num_nodes)}

        # --- Edges ---
        edge_x, edge_y, edge_z = [], [], []
        for u, v in graph_nx.edges():
            x0, y0, z0 = pos_3d[u]
            x1, y1, z1 = pos_3d[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_z += [z0, z1, None]

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color=color, width=1),
            hoverinfo='none',
            showlegend=False
        ))

        # --- Nodes ---
        node_x = [pos_3d[n][0] for n in graph_nx.nodes()]
        node_y = [pos_3d[n][1] for n in graph_nx.nodes()]
        node_z = [pos_3d[n][2] for n in graph_nx.nodes()]

        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(size=3, color=color, opacity=0.9),
            hovertext=[str(n) for n in graph_nx.nodes()],
            hoverinfo='text',
            name=titles[i],  # shows in legend
            showlegend=True
        ))

    # Bare canvas: no title, axes or margins, transparent background, so the
    # export drops straight into a figure.
    _blank_axis = dict(
        title='', showbackground=False, showgrid=False,
        zeroline=False, showticklabels=False, showspikes=False,
        visible=False
    )
    fig.update_layout(
        title='',
        width=1920,
        height=1080,
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=_blank_axis,
            yaxis=_blank_axis,
            zaxis=_blank_axis,
            aspectmode='data'
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0)
    )


    # ParaView values
    position    = np.array([0.0855408, 0.0830408, 0.0747909])
    focal_point = np.array([0.0,      -0.0025,   -0.0107499])
    view_up     = np.array([-0.408248, 0.816497, -0.408248])

    # Direction and distance
    direction = position - focal_point
    distance  = np.linalg.norm(direction)
    eye_norm  = direction / distance  # unit vector, then scale for Plotly

    scale = 2.0  # Plotly's default "comfortable" distance

    camera = dict(
        up=dict(x=view_up[0],        y=view_up[1],        z=view_up[2]),
        center=dict(x=0, y=0, z=0),  # Plotly centers on data automatically
        eye=dict(
            x=eye_norm[0] * scale,
            y=eye_norm[1] * scale,
            z=eye_norm[2] * scale
        )
    )

    fig.update_layout(scene_camera=camera)

    if savepath:
        fig.write_image(savepath)
    else:
        fig.show()