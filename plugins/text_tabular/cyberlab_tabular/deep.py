"""Reviewed deep-learning templates for tabular intrusion-detection data (PyTorch).

Each model is wrapped in a small scikit-learn style estimator (`fit`, `predict`,
`predict_proba`), so the trusted runner, the validation-only tuner and the statistics treat it
exactly like the classical templates. Platform-owned code: agents choose a template and
parameters from the allowlist; they never write the code that runs on the GPU.

Models
- ResNet-MLP and FT-Transformer follow Gorishniy et al., "Revisiting Deep Learning Models for
  Tabular Data" (NeurIPS 2021), the standard strong deep baselines for tabular data.
- TabNet follows Arik and Pfister (AAAI 2021) in a compact form: sequential attentive steps with
  sparse feature masks (sparsemax), shared across steps.
- 1D-CNN treats the feature vector as a sequence, as many intrusion-detection papers do.

Training: quantile-normalised inputs, AdamW, cosine learning-rate decay, optional class weights,
early stopping on a 10% holdout drawn from the TRAINING data only (the protocol's validation
split stays reserved for tuning, and the test split is never seen), mixed precision on the GPU.
"""

from __future__ import annotations

# torch.nn modules return untyped tensors to mypy
# mypy: disable-error-code="no-any-return"
import math
from typing import Any, cast

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn

# ---------------------------------------------------------------------------- architectures


class _ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(d),
            nn.Linear(d, 2 * d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResNetMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, d: int, n_blocks: int, dropout: float) -> None:
        super().__init__()
        self.inp = nn.Linear(n_in, d)
        self.blocks = nn.Sequential(*[_ResBlock(d, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.inp(x)))


class FTTransformer(nn.Module):
    """Feature tokenizer (one learned embedding per numeric feature) + Transformer + [CLS]."""

    def __init__(
        self, n_in: int, n_out: int, d_token: int, n_blocks: int, n_heads: int, dropout: float
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_in, d_token))
        self.bias = nn.Parameter(torch.empty(n_in, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        for p in (self.weight, self.bias, self.cls):
            nn.init.normal_(p, std=1 / math.sqrt(d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=int(d_token * 4 / 3) * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(), nn.Linear(d_token, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # (batch, features, d_token)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0])


class CNN1D(nn.Module):
    def __init__(
        self, n_in: int, n_out: int, channels: int, n_layers: int, kernel_size: int, dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = 1
        for i in range(n_layers):
            c_out = channels * (2 ** min(i, 2))
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(c_in, n_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.unsqueeze(1)))


def _sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Sparsemax over the last dimension (Martins and Astudillo 2016)."""
    zs, _ = torch.sort(z, dim=-1, descending=True)
    rng = torch.arange(1, z.shape[-1] + 1, device=z.device, dtype=z.dtype)
    cum = zs.cumsum(-1)
    support = 1 + rng * zs > cum
    k = support.sum(dim=-1, keepdim=True).clamp(min=1)
    tau = (cum.gather(-1, k - 1) - 1) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


class _GLU(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_in, 2 * d_out)
        self.bn = nn.BatchNorm1d(2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.bn(self.fc(x)).chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class TabNet(nn.Module):
    """Compact TabNet: shared feature transformer, per-step attentive masks with prior scaling."""

    def __init__(self, n_in: int, n_out: int, n_d: int, n_steps: int, gamma: float) -> None:
        super().__init__()
        self.n_d, self.n_steps, self.gamma = n_d, n_steps, gamma
        self.bn = nn.BatchNorm1d(n_in)
        self.shared = nn.ModuleList([_GLU(n_in, 2 * n_d), _GLU(2 * n_d, 2 * n_d)])
        self.steps = nn.ModuleList([_GLU(2 * n_d, 2 * n_d) for _ in range(n_steps + 1)])
        self.attn = nn.ModuleList([nn.Linear(n_d, n_in) for _ in range(n_steps)])
        self.attn_bn = nn.ModuleList([nn.BatchNorm1d(n_in) for _ in range(n_steps)])
        self.head = nn.Linear(n_d, n_out)
        self.sparsity = torch.tensor(0.0)

    def _transform(self, x: torch.Tensor, i: int) -> torch.Tensor:
        h = self.shared[0](x)
        h = (h + self.shared[1](h)) * math.sqrt(0.5)
        return (h + self.steps[i](h)) * math.sqrt(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        prior = torch.ones_like(x)
        a = self._transform(x, 0)[:, self.n_d :]
        out = torch.zeros(len(x), self.n_d, device=x.device, dtype=x.dtype)
        entropy = torch.zeros((), device=x.device)
        for step in range(self.n_steps):
            mask = _sparsemax(self.attn_bn[step](self.attn[step](a)) * prior)
            prior = prior * (self.gamma - mask)
            entropy = entropy + (-mask * torch.log(mask + 1e-10)).sum(-1).mean()
            h = self._transform(x * mask, step + 1)
            out = out + torch.relu(h[:, : self.n_d])
            a = h[:, self.n_d :]
        self.sparsity = entropy / self.n_steps
        return self.head(out)


# ---------------------------------------------------------------------------- estimator


class TorchTabularClassifier:
    """scikit-learn style wrapper: fit(X, y) with integer labels, predict, predict_proba."""

    def __init__(
        self,
        arch: str,
        params: dict[str, Any],
        seed: int,
        gpu: bool,
        *,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 30,
        batch_size: int = 1024,
        class_weight: str = "none",
        patience: int = 5,
    ) -> None:
        self.arch, self.params, self.seed = arch, params, seed
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.lr, self.weight_decay, self.epochs = lr, weight_decay, epochs
        self.batch_size, self.class_weight, self.patience = batch_size, class_weight, patience
        self.history: list[dict[str, float]] = []

    def _build(self, n_in: int, n_out: int) -> nn.Module:
        p = self.params
        if self.arch == "resnet_mlp":
            return ResNetMLP(n_in, n_out, p["d"], p["n_blocks"], p["dropout"])
        if self.arch == "ft_transformer":
            return FTTransformer(
                n_in, n_out, p["d_token"], p["n_blocks"], p["n_heads"], p["dropout"]
            )
        if self.arch == "cnn1d":
            return CNN1D(n_in, n_out, p["channels"], p["n_layers"], p["kernel_size"], p["dropout"])
        if self.arch == "tabnet":
            return TabNet(n_in, n_out, p["n_d"], p["n_steps"], p["gamma"])
        raise ValueError(f"unknown deep architecture {self.arch}")

    def fit(self, x: np.ndarray, y: np.ndarray) -> TorchTabularClassifier:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_classes_ = int(y.max()) + 1
        self.scaler_ = QuantileTransformer(
            output_distribution="normal", n_quantiles=min(1000, len(x)), random_state=self.seed
        ).fit(x)
        xs = self.scaler_.transform(x).astype(np.float32)
        # early-stopping holdout from the TRAINING rows only, stratified per class
        hold = np.zeros(len(y), dtype=bool)
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            k = int(len(idx) * 0.1) if len(idx) >= 10 else 0
            hold[rng.choice(idx, size=k, replace=False)] = True
        xt, yt, xh, yh = xs[~hold], y[~hold], xs[hold], y[hold]
        self.model_ = self._build(xs.shape[1], self.n_classes_).to(self.device)
        weight = None
        if self.class_weight == "balanced":
            counts = np.bincount(yt, minlength=self.n_classes_).astype(np.float32)
            w = np.where(counts > 0, len(yt) / (self.n_classes_ * np.maximum(counts, 1)), 0.0)
            weight = torch.tensor(np.sqrt(w), dtype=torch.float32, device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.epochs))
        amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        xt_t = torch.from_numpy(xt)
        yt_t = torch.from_numpy(yt.astype(np.int64))
        best, best_state, bad = math.inf, None, 0
        # at least ~20 optimiser steps per epoch, so small datasets still train
        bs = max(32, min(self.batch_size, len(xt) // 20 or 32))
        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.from_numpy(rng.permutation(len(xt)))
            total = 0.0
            for i in range(0, len(perm), bs):
                b = perm[i : i + bs]
                if len(b) < 2:  # BatchNorm needs two rows
                    continue
                xb, yb = xt_t[b].to(self.device), yt_t[b].to(self.device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(self.device.type, enabled=amp):
                    loss = loss_fn(self.model_(xb), yb)
                    if self.arch == "tabnet":
                        loss = loss + 1e-3 * cast(torch.Tensor, self.model_.sparsity)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += float(loss.detach()) * len(b)
            sched.step()
            val = self._loss(xh, yh, loss_fn) if len(xh) else total / len(xt)
            self.history.append(
                {"epoch": epoch + 1, "train_loss": total / len(xt), "val_loss": val}
            )
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.epochs_run_ = len(self.history)
        return self

    @torch.no_grad()
    def _logits(self, xs: np.ndarray) -> np.ndarray:
        self.model_.eval()
        out = []
        for i in range(0, len(xs), 8192):
            xb = torch.from_numpy(xs[i : i + 8192]).to(self.device)
            out.append(self.model_(xb).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.n_classes_), dtype=np.float32)

    def _loss(self, xs: np.ndarray, y: np.ndarray, loss_fn: nn.Module) -> float:
        logits = torch.from_numpy(self._logits(xs))
        w = getattr(loss_fn, "weight", None)
        return float(
            nn.functional.cross_entropy(
                logits,
                torch.from_numpy(y.astype(np.int64)),
                weight=w.cpu() if w is not None else None,
            )
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        xs = self.scaler_.transform(x).astype(np.float32)
        return torch.softmax(torch.from_numpy(self._logits(xs)), dim=-1).numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)
