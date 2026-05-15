import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go

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

    colors = ['steelblue', 'mediumseagreen', 'mediumpurple', 'orange']

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

    fig.update_layout(
        title='3D Graph',
        width=1920,
        height=1080,
        scene=dict(
            xaxis=dict(title='X', showbackground=False),
            yaxis=dict(title='Y', showbackground=False),
            zaxis=dict(title='Z', showbackground=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-5., y=0., z=4)
)
    fig.update_layout(scene_camera=camera)

    if savepath:
        fig.write_image(savepath)
    else:
        fig.show()