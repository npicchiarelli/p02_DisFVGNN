"""Read the per-edge messages m_ij out of a trained FVSurrogate.

The messages are what an FV comparison needs. `FiniteVolumeGraphNet` sums them
at the receiving node the way a finite-volume scheme sums face fluxes over a
cell, so m_ij is the network's learned stand-in for the flux through the face
that edge carries — but `propagate` only ever returns the *updated node
features*, so m_ij is discarded before `forward` returns.

Extraction goes through PyG's `register_message_forward_hook` rather than a
`return_messages` flag on `FiniteVolumeGraphNet.forward`. The hook leaves the
train/inference path and the state_dict untouched, so every checkpoint already
under processed_data/*/checkpoints/model.pt can be probed as-is.

ORIENTATION — the one thing to get right before comparing anything.
PyG's default source_to_target flow aggregates at edge_index[1], so inside
`message()` x_i is the RECEIVER and x_j the SENDER. build_static_graph orients
that edge's stored Sf source -> target, i.e. INTO the receiver, precisely so
that summing messages at a node mirrors an FV flux balance over its faces.
`fv_flux` below returns the flux under that same convention: positive heats the
receiver, and sum over the in-edges of node i gives V_i dT_i/dt.

Usage
-----
    probe = MessageProbe(model)
    with probe:
        T_next = model(data)

    m = probe.messages[0]      # (E, msg_dim) — layer 0's per-edge message
    M = probe.aggregated[0]    # (N, msg_dim) — sum_j m_ij at each node

    F = fv_flux(static_graph, T_phys, DT=4e-5)   # (E,) reference FV flux

Note that `data.edge_attr` fed to the model is NORMALIZED, while `fv_flux` wants
the raw geometry — pass `static_graph.edge_attr`, not `data.edge_attr`.
"""

import torch
from torch_geometric.nn import MessagePassing


# Column layout of static_graph.edge_attr, as packed by build_static_graph.
EDGE_ATTR_COLS = {
    "dist_vec": slice(0, 3),
    "dist_norm": 3,
    "Sf": slice(4, 7),
    "magSf": 7,
    "skewness": 8,
    "non_orth": 9,          # cos(Sf, d), NOT an angle
}


class MessageProbe:
    """Capture m_ij and its aggregation for every MessagePassing layer in `model`.

    Layers are indexed in `model.modules()` order, which for FVSurrogate is the
    mp_layers order (layer 0 first).

    Args:
        model       : any nn.Module containing MessagePassing layers
        keep_inputs : also record the message MLP's inputs (x_i, x_j, edge_attr)
        to_cpu      : move captured tensors off the GPU as they are recorded
        detach      : detach from the autograd graph (leave True unless you want
                      to backprop *through* a captured message)
    """

    def __init__(self, model, keep_inputs: bool = False,
                 to_cpu: bool = True, detach: bool = True):
        self.layers = [m for m in model.modules() if isinstance(m, MessagePassing)]
        if not self.layers:
            raise ValueError("no MessagePassing layer found in model")
        self.keep_inputs = keep_inputs
        self.to_cpu = to_cpu
        self.detach = detach
        self._handles = []
        self.clear()

    # ── recording ──────────────────────────────────────────────────────────
    def clear(self):
        n = len(self.layers)
        self.messages_all = [[] for _ in range(n)]
        self.aggregated_all = [[] for _ in range(n)]
        self.inputs_all = [[] for _ in range(n)]

    def _keep(self, t):
        if self.detach:
            t = t.detach()
        return t.cpu() if self.to_cpu else t

    def _make_msg_hook(self, i):
        def hook(module, inputs, output):
            self.messages_all[i].append(self._keep(output))
            if self.keep_inputs:
                (msg_kwargs,) = inputs
                self.inputs_all[i].append(
                    {k: self._keep(v) for k, v in msg_kwargs.items()
                     if torch.is_tensor(v)}
                )
        return hook

    def _make_aggr_hook(self, i):
        def hook(module, inputs, output):
            self.aggregated_all[i].append(self._keep(output))
        return hook

    def __enter__(self):
        self.clear()
        for i, layer in enumerate(self.layers):
            # decomposed_layers > 1 would split one propagate into several
            # message() calls over feature chunks; the probe would then record
            # partial-width messages per call instead of one full message.
            if getattr(layer, "decomposed_layers", 1) != 1:
                raise RuntimeError(
                    f"layer {i} has decomposed_layers="
                    f"{layer.decomposed_layers}; set it to 1 to probe messages"
                )
            self._handles.append(layer.register_message_forward_hook(
                self._make_msg_hook(i)))
            self._handles.append(layer.register_aggregate_forward_hook(
                self._make_aggr_hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # ── access ─────────────────────────────────────────────────────────────
    @property
    def messages(self):
        """Per-layer message of the MOST RECENT forward pass: list of (E, msg_dim)."""
        return [calls[-1] for calls in self.messages_all]

    @property
    def aggregated(self):
        """Per-layer aggregated message of the most recent pass: list of (N, msg_dim)."""
        return [calls[-1] for calls in self.aggregated_all]

    def stacked_messages(self, layer: int = 0) -> torch.Tensor:
        """(n_calls, E, msg_dim) — every forward pass recorded since __enter__.

        Use this over a rollout / a loop of timesteps to get m_ij(t).
        """
        return torch.stack(self.messages_all[layer], dim=0)

    def stacked_aggregated(self, layer: int = 0) -> torch.Tensor:
        """(n_calls, N, msg_dim) — the aggregated counterpart."""
        return torch.stack(self.aggregated_all[layer], dim=0)

    def __repr__(self):
        n_calls = len(self.messages_all[0])
        shapes = [tuple(c[-1].shape) if c else None for c in self.messages_all]
        return (f"MessageProbe(layers={len(self.layers)}, calls={n_calls}, "
                f"message_shapes={shapes})")


# ── the reference the messages get compared against ────────────────────────

def fv_flux(edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            T: torch.Tensor,
            DT: float,
            nonorth_correction: bool = True) -> torch.Tensor:
    """Orthogonal FV diffusive flux through each directed edge, INTO its receiver.

    This is the implicit part of what `laplacianFoam` assembles for
    `laplacian(DT,T)` with `Gauss linear corrected`:

        F_(j->i) = DT * |Sf| * (T_j - T_i) / (|d| * max(cos, 0.05))

    where cos = cos(Sf, d) is the non-orthogonality feature (edge_attr column 9)
    and the 0.05 floor is OpenFOAM's own limiter in `nonOrthDeltaCoeffs`. Summed
    over the in-edges of node i this is V_i dT_i/dt, so it is directly
    comparable to sum_j m_ij.

    The EXPLICIT non-orthogonal correction k_f . (grad T)_f is NOT included: it
    needs a reconstructed cell gradient, not a per-edge quantity. HOW BIG that
    omission is, measured on the flange (t=100..140, per-cell least-squares fit
    of V_i in  V_i dT_i/dt = sum_j F_ij):

        worst face cos of cell   cells   fits giving V_i < 0 (impossible)
        > 0.999                   134     0 %      V_i median 2.1e-09
        0.99 - 0.999              194     2 %
        0.95 - 0.99              1018    19 %
        < 0.95                   1510    31 %

    i.e. this flux is EXACT on the orthogonal part of the flange and degrades
    monotonically with non-orthogonality; the mesh reaches cos = 0.72. The
    balance holds to R^2 ~ 0.985 mesh-wide, and the implied cell volumes are the
    right size (~2e-9 m^3), so the orientation and scaling are right — but do
    not read a message/flux mismatch on a strongly non-orthogonal cell as the
    network's failure. Restrict to high-cos edges when that distinction matters.

    The geometry underneath is exact: cos(Sf, d) > 0 on all 32096 flange edges
    (min 0.72), Sf_ij = -Sf_ji to machine zero, and the stored non_orth column
    matches the recomputed cos to 4e-5 on internal edges.

    Args:
        edge_index : (2, E)   — static_graph.edge_index
        edge_attr  : (E, 10)  — static_graph.edge_attr, RAW (not normalized)
        T          : (N,) or (n_steps, N), physical units
        DT         : diffusivity from constant/transportProperties (flange: 4e-5)
        nonorth_correction : divide by cos as above; False gives the plain
                             1/|d| deltaCoeffs (what a `uncorrected` scheme uses)

    Returns:
        (E,) if T is (N,), else (n_steps, E)
    """
    if edge_attr.shape[1] < 10:
        raise ValueError(
            f"fv_flux needs the full 10-column edge_attr, got {edge_attr.shape[1]}. "
            "The *_nofv runs slice it to 4 columns for the MODEL — build the flux "
            "from the unsliced static_graph.edge_attr."
        )

    src, dst = edge_index[0], edge_index[1]
    magSf = edge_attr[:, EDGE_ATTR_COLS["magSf"]].double()
    magd = edge_attr[:, EDGE_ATTR_COLS["dist_norm"]].double()
    cos = edge_attr[:, EDGE_ATTR_COLS["non_orth"]].double()

    delta_coeffs = 1.0 / (magd * cos.clamp(min=0.05)) if nonorth_correction \
        else 1.0 / magd

    # accumulate in float64 (magSf ~ 1e-6 on the flange, so float32 loses bits),
    # then hand the result back in the field's own dtype
    dT = T.double()[..., src] - T.double()[..., dst]   # T_sender - T_receiver
    return (DT * magSf * delta_coeffs * dT).to(T.dtype)


def paired_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """For each directed edge, the index of its reverse (j->i), or -1 if absent.

    Every mesh face becomes two opposite directed edges, so this pairs an edge
    with the other half of its face. A true FV flux is exactly antisymmetric
    across that pair (F_ij = -F_ji, which is what makes the scheme
    conservative); whether m_ij is, is a direct test of how FV-like the learned
    message is.
    """
    E = edge_index.shape[1]
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    lookup = {(s, d): e for e, (s, d) in enumerate(zip(src, dst))}
    return torch.tensor(
        [lookup.get((d, s), -1) for s, d in zip(src, dst)], dtype=torch.long
    )


def load_cell_gradient(case_dir: str, times) -> torch.Tensor:
    """(n_times, n_cells, 3) cell-centred grad(T), from OpenFOAM's own export.

    `laplacianFoam` writes gradTx/gradTy/gradTz (and `flux` = DT * gradT, which
    is the same field scaled — verified to 1.6e-6 relative on the parametric
    cases) at every write time. That is the piece `fv_flux` cannot reconstruct
    from per-edge data, so having it on disk is what makes `fv_flux_corrected`
    below possible without touching OpenFOAM.

    Internal (cell) values only — enough for the correction, which this module
    applies on internal faces.
    """
    import os
    import numpy as np
    from smithers.io.openfoam import field_parser

    out = []
    for t in times:
        comps = [field_parser.parse_internal_field(
            os.path.join(case_dir, str(t), f"gradT{c}")) for c in "xyz"]
        out.append(np.stack(comps, axis=1))
    return torch.from_numpy(np.stack(out, axis=0)).double()


def fv_flux_corrected(edge_index: torch.Tensor,
                      edge_attr: torch.Tensor,
                      T: torch.Tensor,
                      gradT: torch.Tensor,
                      DT: float,
                      n_internal_nodes: int) -> torch.Tensor:
    """`fv_flux` PLUS the explicit non-orthogonal correction — the full flux
    that `Gauss linear corrected` actually assembles.

        F = DT |Sf| [ Delta (T_j - T_i)  +  k . (grad T)_f ]

        Delta = 1 / (|d| max(cos, 0.05))              (as in fv_flux)
        k     = d / (|d| max(cos, 0.05))  -  Sf/|Sf|  (OpenFOAM's
                nonOrthCorrectionVectors, re-expressed for THIS module's
                orientation, where Sf points INTO the receiver)
        (grad T)_f = mean of the two cells' gradients

    k vanishes identically on an orthogonal face, so this reduces to `fv_flux`
    exactly where `fv_flux` was already exact, and only moves the skewed faces.

    TWO APPROXIMATIONS, both second-order and both on the correction term only:
      - (grad T)_f uses a 0.5/0.5 midpoint rather than OpenFOAM's `linear`
        weights, which need face centres the graph does not carry.
      - the correction is applied on INTERNAL faces only; boundary edges keep
        the orthogonal form.

    Args:
        edge_index : (2, E)
        edge_attr  : (E, 10) RAW geometry from build_static_graph
        T          : (N,) or (n_steps, N), physical units
        gradT      : (n_cells, 3) or (n_steps, n_cells, 3), from
                     `load_cell_gradient`; n_cells is the INTERNAL node count
        DT         : diffusivity
        n_internal_nodes : where the boundary nodes start in the graph

    Returns:
        same shape convention as `fv_flux`
    """
    base = fv_flux(edge_index, edge_attr, T, DT, nonorth_correction=True)

    src, dst = edge_index[0], edge_index[1]
    internal = (src < n_internal_nodes) & (dst < n_internal_nodes)

    d = edge_attr[:, EDGE_ATTR_COLS["dist_vec"]].double()
    magd = edge_attr[:, EDGE_ATTR_COLS["dist_norm"]].double()
    Sf = edge_attr[:, EDGE_ATTR_COLS["Sf"]].double()
    magSf = edge_attr[:, EDGE_ATTR_COLS["magSf"]].double()
    cos = edge_attr[:, EDGE_ATTR_COLS["non_orth"]].double()

    delta = 1.0 / (magd * cos.clamp(min=0.05))
    k = d * delta[:, None] - Sf / magSf[:, None]        # (E, 3), zero if orthogonal

    if gradT.dim() == 2:
        gradT = gradT[None]
    gf = 0.5 * (gradT[:, src.clamp(max=n_internal_nodes - 1)]
                + gradT[:, dst.clamp(max=n_internal_nodes - 1)])   # (S, E, 3)

    corr = DT * magSf * (k[None] * gf).sum(-1)          # (S, E)
    corr = corr * internal[None]
    if base.dim() == 1:
        corr = corr[0]
    return (base.double() + corr).to(T.dtype)
