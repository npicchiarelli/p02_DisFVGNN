"""Eigenpairs of the discrete diffusion operator, for node features.

K comes from getter_of.getLaplacian: the homogeneous part of -fvm::laplacian(DT, T)
with T's own boundary conditions and DT as laplacianFoam reads it. The FV matrix is
volume-integrated, so the discrete counterpart of -div(DT grad phi) = lam phi is the
generalized problem

    K phi = lam M phi,    M = diag(V).

With M = V the modes are discrete versions of the continuous eigenfunctions, which
keeps them comparable across meshes whose cell volumes differ by orders of
magnitude; plain K phi = lam phi would give graph-Laplacian modes that bunch up on
the small cells. For laplacianFoam, ddt(T) = laplacian(DT, T), so lam is the decay
rate [1/s] of mode phi in T - T_steady.

numpy/scipy only, so that it can be tested without torch.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def laplacian_eigenpairs(K, V, k: int, rtol: float = 1e-6):
    """The k smallest eigenpairs of K phi = lam diag(V) phi.

    K : (N, N) sparse, symmetric positive (semi)definite
    V : (N,) cell volumes, > 0
    Returns lam (k,) ascending and phi (N, k), V-orthonormal (phi.T @ (V * phi) = I)
    and sign-canonical: the entry of largest |phi_k| is positive. Under any Dirichlet
    BC the first mode has a single sign, so it comes out positive everywhere. For an
    antisymmetric mode the two largest entries are mirror images and the rule picks
    one of them arbitrarily; likewise inside a degenerate eigenspace the basis itself
    is arbitrary. Those are properties of the modes, not of this rule.

    Solved as the symmetric S psi = lam psi, S = M^-1/2 K M^-1/2, phi = M^-1/2 psi,
    by shift-invert at a small negative shift: below every eigenvalue, so the
    eigenvalues nearest to it are the smallest ones, for a positive definite K and
    for a pure-Neumann one with its zero mode alike.
    """
    K = sp.csr_matrix(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64).ravel()
    N = K.shape[0]
    if K.shape != (N, N) or V.shape != (N,):
        raise ValueError(f"K {K.shape} and V {V.shape} do not match")
    if not 0 < k < N - 1:
        raise ValueError(f"k must be in 1..{N - 2}, got {k}")
    if not np.all(V > 0):
        raise ValueError("cell volumes must be > 0")
    scale = abs(K).max()
    if abs(K - K.T).max() > 1e-10 * scale:
        raise ValueError("K is not symmetric")

    d = 1.0 / np.sqrt(V)
    D = sp.diags(d)
    S = (D @ K @ D).tocsc()
    sigma = -1e-8 * abs(S).max()
    lam, psi = eigsh(S, k=k, sigma=sigma, which="LM")

    order = np.argsort(lam)
    lam, psi = lam[order], psi[:, order]
    phi = d[:, None] * psi

    # Guard against a silent eigsh failure: residual relative to K phi
    res = np.linalg.norm(K @ phi - (V[:, None] * phi) * lam, axis=0)
    ref = np.linalg.norm(K @ phi, axis=0) + np.finfo(float).tiny
    if np.any(res > rtol * ref):
        raise RuntimeError(f"eigsh residual too large: max {np.max(res / ref):.2e}")

    idx = np.argmax(np.abs(phi), axis=0)
    phi *= np.sign(phi[idx, np.arange(k)])
    return lam, phi


def boundary_face_values(phi, bnd_faces, face_owner, value_coeffs):
    """phi on boundary faces, from the homogeneous BC: phi_f = c_f * phi[owner(f)].

    phi          : (N_cells, k)
    bnd_faces    : (B,) global face indices, all boundary faces
    face_owner   : (N_faces,) owner cell of every face
    value_coeffs : per patch, in boundary order, getter_of.getBoundaryValueCoeffs.
                   Patches are contiguous in face order after the internal faces,
                   so concatenated they index the boundary faces from nInternalFaces.
    """
    c = np.concatenate([np.asarray(cp, dtype=np.float64).ravel() for cp in value_coeffs])
    bnd_faces = np.asarray(bnd_faces)
    n_internal = len(face_owner) - len(c)
    if np.any(bnd_faces < n_internal):
        raise ValueError("bnd_faces contains internal faces")
    c_f = c[bnd_faces - n_internal]
    if np.any(np.isnan(c_f)):
        raise ValueError("bnd_faces contains faces of a patch without values (empty?)")
    return c_f[:, None] * phi[np.asarray(face_owner)[bnd_faces]]
