import torch
from torch_geometric.loader import DataLoader
from data_preparation.static_graph import build_static_graph
from data_preparation.mesh_dataset import SingleMeshDataset, temporal_split
from data_preparation.normalization import FeatureNormalizer
from data_preparation.field import load_fields
from models.fvgnn import FVSurrogate
from mesh2graph.utils import parse_internal_fields_alltimes, parse_boundary_fields_alltimes
from smithers.io.openfoam import FoamMesh
from sys import exit

torch.default_dtype = torch.float32

# ── 0. Configuration ──────────────────────────────────────────────────

case_dir = "./tests/data/flange"
excluded_patches = ["patch1", "patch3"]


# ── 1. Load your mesh data ──────────────────────────────────────────────────

static_graph = build_static_graph(case_dir, excluded_patches)
T_sequence = load_fields(case_dir, 'T', excluded_patches=excluded_patches)
print("Static graph:", static_graph, static_graph.node_attr.dtype, static_graph.edge_attr.dtype)
print("T sequence shape:", T_sequence.shape, T_sequence.dtype)  # Should be (T, N)
print(static_graph.x)  # Should be (E, F_e)

print(f"Total training samples: {(len(T_sequence) - 5)*len(static_graph.x)}")  # history=5, so we lose the first 5 samples
exit(0)


# ── 2. Split & normalise ────────────────────────────────────────────────────
normalizer = FeatureNormalizer()
history = 5   # use last 5 timesteps to predict next

train_ds, val_ds, test_ds = temporal_split(
    T_sequence, static_graph, normalizer, history=history
)

# normalizer.save("checkpoints/normalizer.pt")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=False,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4)

# ── 3. Model & optimiser ────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

in_node_feat  = history + static_graph.node_attr.shape[1]  # T history + geometry
in_edge_feat  = static_graph.edge_attr.shape[1]            # 10

model = FVSurrogate(
    in_node_feat=in_node_feat,
    in_edge_feat=in_edge_feat,
    hidden_dim=64,
    out_dim=1,
    n_mp_layers=4,         # message passing depth — think of it as stencil reach
).to(device)

print(f"Model initialized with parameters: {sum(p.numel() for p in model.parameters()):4e}")

exit(0)


optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

# ── 4. Training loop ────────────────────────────────────────────────────────
for epoch in range(500):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)                    # (batch_N,)
        loss = torch.nn.functional.mse_loss(pred, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            val_loss += torch.nn.functional.mse_loss(pred, batch.y).item()

    val_loss /= len(val_loader)
    # scheduler.step(val_loss)
    print(f"Epoch {epoch:4d} | train {total_loss/len(train_loader):.4e} | val {val_loss:.4e}")