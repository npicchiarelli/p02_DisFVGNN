"""Sparsity penalties on the per-edge message m_ij of an FVSurrogate.

The goal is a message with few active channels, small enough to regress
symbolically (Cranmer et al. 2020) and to compare against the FV face flux of
models/message_probe.py. Both penalties act on the message m of the current
training batch, shape (E, msg_dim):

  l1     mean over edges and channels of |m|       (Cranmer et al.'s baseline)
  hoyer  PR = (sum_c sigma_c)^2 / sum_c sigma_c^2, sigma_c the std of channel c
         over the edges: the effective number of active channels. k equally
         active channels and the rest silent give PR = k exactly.

and enter the loss as

  loss = mse + alpha * ramp(epoch) * mse_ema * pen / ref

mse_ema is a moving average of the batch MSE, so the penalty keeps a fixed
exchange rate with the data term however far the MSE falls: alpha is the
fractional MSE increase accepted per unit of penalty removed. For l1 the unit is
the L1 at the end of warmup (ref, averaged over the last warmup epoch); for
hoyer it is one effective channel (ref = 1). ramp is 0 for `warmup` epochs,
then rises linearly to 1 over `ramp` epochs.

The rescaling loophole
----------------------
msg_fnc ends in a Linear and node_fnc starts with one, so scaling channel c of
the message by s_c and column c of node_fnc[0]'s message block by 1/s_c leaves
every prediction unchanged. Both penalties fall along that direction — Hoyer
too, since it only ignores a scale COMMON to all channels — so the network
could satisfy either one by shrinking channels without dropping information.
With a penalty on, `after_step` removes that freedom: after every optimizer
step it moves each channel along the symmetry until its node_fnc[0] column has
norm c0. Predictions do not change, and sigma_c becomes an honest measure of how
much the node update uses channel c.

Usage
-----
    reg = MessageRegularizer(model, kind="hoyer", alpha=1e-2)
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()
            mse = F.mse_loss(model(batch), batch.y)
            reg.loss(mse, epoch).backward()
            optimizer.step()
            reg.after_step()
        reg.end_epoch(epoch)
        val_mse = ...
        if reg.ramp(epoch) == 1.0:
            scheduler.step(reg.plateau_metric(val_mse))
    reg.save(path)
"""

import math

import numpy as np
import torch
import torch.nn as nn


def participation_ratio(sigma: torch.Tensor) -> torch.Tensor:
    """(sum sigma)^2 / sum sigma^2 — effective number of active channels, in [1, len(sigma)]."""
    return sigma.sum().pow(2) / sigma.pow(2).sum().clamp(min=torch.finfo(sigma.dtype).tiny)


class MessageRegularizer:
    """Message penalty, per-channel scale fix and per-epoch message statistics
    for an FVSurrogate with a single message-passing layer.

    Constructing it registers a message hook on that layer (it records
    training-mode passes only) and, with a penalty on, applies the scale fix
    once so it holds from the first step. Neither changes a prediction.

    kind "none" trains exactly as without the regularizer and only logs. "l1"
    or "hoyer" with alpha=0 fixes the scale and logs the penalty without
    training on it: the control for a penalised run.

    Per-epoch log (`log`, written by `save`):
      epoch, ramp, mse_ema, lam_eff  (= alpha * ramp * mse_ema / ref)
      pen        mean over the epoch's batches of the penalty as the loss sees it
      pr         participation ratio of sigma, pooled over all the epoch's edges
      mean_abs   mean |m|, pooled likewise (the l1 penalty for any kind)
      mu, sigma  (msg_dim,) per-channel mean and std, pooled likewise, UNSORTED;
                 mu is what a mean-ablation of a channel substitutes
      col_norm   (msg_dim,) node_fnc[0] message-column norms (all c0 when fixed)
      row_norm   (msg_dim,) msg_fnc[-1] row norms, weight and bias together
      grad_data, grad_pen   norms of the MSE and penalty gradients w.r.t.
                 msg_fnc[-1].weight on the epoch's first penalised batch

    Args:
        model  : FVSurrogate with n_mp_layers=1 (and, with a penalty, no
                 message LayerNorm)
        kind   : "none" | "l1" | "hoyer"
        alpha  : fractional MSE increase accepted per unit of penalty removed
        warmup : epochs with the penalty off
        ramp   : epochs over which it then rises linearly to full strength
        ema    : decay of the MSE moving average, per step
        eps    : added to each channel's variance before the sqrt in hoyer,
                 whose gradient is infinite at zero variance
    """

    KINDS = ("none", "l1", "hoyer")
    LOG_KEYS = ("epoch", "ramp", "mse_ema", "pen", "lam_eff", "pr", "mean_abs",
                "mu", "sigma", "col_norm", "row_norm", "grad_data", "grad_pen")

    def __init__(self, model, kind: str = "none", alpha: float = 0.0,
                 warmup: int = 20, ramp: int = 10, ema: float = 0.99,
                 eps: float = 1e-12):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        if kind == "none" and alpha != 0.0:
            raise ValueError(f"kind 'none' has no penalty to weight, got alpha={alpha}")
        if len(model.mp_layers) != 1:
            raise ValueError(
                f"written for n_mp_layers=1, got {len(model.mp_layers)}: the "
                "penalty and the scale fix act on a single message")
        layer = model.mp_layers[0]
        if kind != "none" and not isinstance(layer.msg_norm, nn.Identity):
            raise ValueError(
                "the scale fix needs msg_fnc[-1] to produce the message, but a "
                "message LayerNorm follows it: build the model with layer_norm=False")
        if kind == "l1" and warmup < 1:
            raise ValueError("l1 measures its reference over the last warmup "
                             f"epoch, so warmup must be >= 1, got {warmup}")

        self.layer = layer
        self.last = layer.msg_fnc[-1]
        self.msg_dim = self.last.out_features
        # update() feeds node_fnc [x_i | M_i], so the message block is the tail
        self.n_f = layer.node_fnc[0].in_features - self.msg_dim
        self.kind = kind
        self.alpha = float(alpha)
        self.warmup = warmup
        self.ramp_len = ramp
        self.ema = ema
        self.eps = eps

        # Target norm of every message column of node_fnc[0]: their mean at
        # init, so the message keeps roughly its initial scale.
        with torch.no_grad():
            self.c0 = self._msg_columns().norm(dim=0).mean().item()
        if kind != "none":
            self._rebalance()

        self.mse_ema = None
        self.ref = 1.0 if kind == "hoyer" else None   # l1: set at the end of warmup
        self._msg = None
        self._handle = layer.register_message_forward_hook(self._capture)
        self.log = {k: [] for k in self.LOG_KEYS}
        self._new_epoch()

    # ── training-loop API ──────────────────────────────────────────────────
    def ramp(self, epoch: int) -> float:
        if epoch < self.warmup:
            return 0.0
        if self.ramp_len <= 0:
            return 1.0
        return min(1.0, (epoch - self.warmup + 1) / self.ramp_len)

    def penalty(self, msg: torch.Tensor) -> torch.Tensor:
        if self.kind == "l1":
            return msg.abs().mean()
        if self.kind == "hoyer":
            return participation_ratio((msg.var(dim=0) + self.eps).sqrt())
        raise ValueError("kind 'none' has no penalty")

    def loss(self, mse: torch.Tensor, epoch: int) -> torch.Tensor:
        """The MSE plus the penalty on the message of the forward pass just run."""
        m = mse.item()
        self.mse_ema = m if self.mse_ema is None else self.ema * self.mse_ema + (1.0 - self.ema) * m
        if self.kind == "none":
            return mse
        if self._msg is None:
            raise RuntimeError("no message captured: call loss() after a "
                               "forward pass in training mode")

        scale = self.alpha * self.ramp(epoch)
        if scale == 0.0:
            # warmup, or the alpha=0 control: logged (and l1's ref measured),
            # not trained on
            with torch.no_grad():
                self._pen_sum += self.penalty(self._msg).item()
            self._pen_n += 1
            return mse

        if self.ref is None:
            raise RuntimeError("l1 reference not set: call end_epoch() at the "
                               "end of every epoch")
        pen = self.penalty(self._msg)
        term = (scale * self.mse_ema / self.ref) * pen
        if self._check_grads:
            # once per epoch: how hard each term pulls on the message head
            w = self.last.weight
            self._grad_data = torch.autograd.grad(mse, w, retain_graph=True)[0].norm().item()
            self._grad_pen = torch.autograd.grad(term, w, retain_graph=True)[0].norm().item()
            self._check_grads = False
        self._pen_sum += pen.item()
        self._pen_n += 1
        return mse + term

    @torch.no_grad()
    def after_step(self):
        """Call right after optimizer.step(): records the batch's message
        statistics, releases the message, and with a penalty on fixes the
        per-channel scale."""
        msg, self._msg = self._msg, None
        if msg is not None:
            if self._sum is None:
                self._sum, self._sq, self._abs = (
                    torch.zeros(self.msg_dim, dtype=torch.float64, device=msg.device)
                    for _ in range(3))
            # in chunks: a full-size temporary of a multi-million-edge
            # message would cost as much memory as the message itself
            for chunk in msg.detach().split(1 << 20):
                self._sum += chunk.sum(0, dtype=torch.float64)
                self._sq += chunk.pow(2).sum(0, dtype=torch.float64)
                self._abs += chunk.abs().sum(0, dtype=torch.float64)
            self._n += msg.shape[0]
        if self.kind != "none":
            self._rebalance()

    @torch.no_grad()
    def end_epoch(self, epoch: int) -> dict:
        """Close the epoch: append its log entry and return it."""
        if self._n > 0:
            mean = self._sum / self._n
            sigma_t = (self._sq / self._n - mean.pow(2)).clamp(min=0.0).sqrt()
            pr = participation_ratio(sigma_t).item()
            mean_abs = (self._abs.sum() / (self._n * self.msg_dim)).item()
            mu, sigma = mean.cpu().numpy(), sigma_t.cpu().numpy()
        else:
            pr = mean_abs = float("nan")
            mu = sigma = np.full(self.msg_dim, np.nan)

        pen = self._pen_sum / self._pen_n if self._pen_n else float("nan")
        if self.kind == "l1" and epoch == self.warmup - 1:
            self.ref = pen
        ramp = self.ramp(epoch)
        mse_ema = float("nan") if self.mse_ema is None else self.mse_ema
        lam_eff = (self.alpha * ramp * mse_ema / self.ref
                   if self.kind != "none" and ramp > 0.0 else 0.0)

        entry = dict(
            epoch=epoch, ramp=ramp, mse_ema=mse_ema, pen=pen, lam_eff=lam_eff,
            pr=pr, mean_abs=mean_abs, mu=mu, sigma=sigma,
            col_norm=self._msg_columns().norm(dim=0).cpu().numpy(),
            row_norm=torch.cat([self.last.weight, self.last.bias[:, None]], 1)
                          .norm(dim=1).cpu().numpy(),
            grad_data=self._grad_data, grad_pen=self._grad_pen,
        )
        for k, v in entry.items():
            self.log[k].append(v)
        self._new_epoch()
        return entry

    def plateau_metric(self, val_mse: float) -> float:
        """What ReduceLROnPlateau should watch: val_mse * exp(alpha * ramp * pen / ref),
        with ramp and pen from the last epoch closed by `end_epoch`.

        The loss gradient is ~ mse * grad(log mse + alpha * ramp * pen / ref),
        so this is the quantity actually being minimised: a run trading MSE
        for sparsity at the rate alpha allows still counts as improving. The
        bare val MSE would not, and the scheduler would cut the LR while the
        penalty is doing its job. Equals val_mse for kind "none" and alpha = 0.

        Step the scheduler only once ramp(epoch) == 1: while the ramp rises the
        metric changes definition every epoch, and each rise reads as a bad one.
        """
        if self.kind == "none" or self.alpha == 0.0:
            return val_mse
        if not self.log["pen"]:
            raise RuntimeError("no epoch closed yet: call end_epoch() before plateau_metric()")
        return val_mse * math.exp(self.alpha * self.log["ramp"][-1]
                                  * self.log["pen"][-1] / self.ref)

    def save(self, path):
        """Write the per-epoch log and the configuration to an .npz."""
        np.savez(
            path,
            **{k: np.asarray(v) for k, v in self.log.items()},
            kind=self.kind, alpha=self.alpha, warmup=self.warmup,
            ramp_len=self.ramp_len, ema=self.ema, c0=self.c0,
            ref=np.nan if self.ref is None else self.ref,
        )

    def remove(self):
        """Detach the message hook."""
        self._handle.remove()

    # ── internals ──────────────────────────────────────────────────────────
    def _capture(self, module, inputs, output):
        # Training passes only: validation and rollout run in eval mode, and
        # keeping their messages would only hold memory.
        if module.training:
            self._msg = output

    def _msg_columns(self) -> torch.Tensor:
        """(hidden, msg_dim) view of node_fnc[0].weight acting on M_i."""
        return self.layer.node_fnc[0].weight[:, self.n_f:]

    @torch.no_grad()
    def _rebalance(self):
        # Channel c: column c of node_fnc[0] divided by r_c, row c of
        # msg_fnc[-1] (weight and bias) multiplied by r_c. M_i is linear in m
        # and node_fnc[0] is linear in M_i, so predictions do not move — only
        # where the channel's scale sits.
        W = self._msg_columns()
        r = (W.norm(dim=0) / self.c0).clamp(min=1e-12)
        W.div_(r)
        self.last.weight.mul_(r[:, None])
        self.last.bias.mul_(r)

    def _new_epoch(self):
        self._sum = self._sq = self._abs = None
        self._n = 0
        self._pen_sum, self._pen_n = 0.0, 0
        self._grad_data = self._grad_pen = float("nan")
        self._check_grads = True
