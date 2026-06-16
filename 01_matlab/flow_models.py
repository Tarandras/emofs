import numpy as np
import torch
import torch.nn as nn

from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)

torch.set_default_dtype(torch.float32)

# ============================================================
# MLP (configurable size)
# ============================================================
class SafeMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden=128, depth=2):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_features, hidden))
        layers.append(nn.ReLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x, context=None):
        return self.net(x)


def create_mlp_factory(hidden, depth):
    def create_mlp(in_features, out_features):
        return SafeMLP(in_features, out_features, hidden, depth)
    return create_mlp


# ============================================================
# MASK
# ============================================================
def alternating_mask(dim, layer_idx):
    mask = (torch.arange(dim) + layer_idx) % 2
    return mask.float()


# ============================================================
# BUILDERS (mode-aware)
# ============================================================
def build_realnvp(dim, layers, hidden, depth):
    transforms = []
    create_mlp = create_mlp_factory(hidden, depth)

    for i in range(layers):
        mask = alternating_mask(dim, i)

        transforms.append(
            AffineCouplingTransform(
                mask=mask,
                transform_net_create_fn=create_mlp
            )
        )
        transforms.append(ReversePermutation(features=dim))

    return Flow(CompositeTransform(transforms), StandardNormal([dim]))


def build_maf(dim, layers, hidden, blocks):
    transforms = []

    for _ in range(layers):
        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden,
                num_blocks=blocks
            )
        )
        transforms.append(ReversePermutation(features=dim))

    return Flow(CompositeTransform(transforms), StandardNormal([dim]))


def build_nsf(dim, layers, hidden, blocks, bins):
    transforms = []

    for _ in range(layers):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=dim,
                hidden_features=hidden,
                num_blocks=blocks,
                num_bins=bins,
                tails="linear"
            )
        )
        transforms.append(ReversePermutation(features=dim))

    return Flow(CompositeTransform(transforms), StandardNormal([dim]))


# ============================================================
# FAST vs ACCURATE CONFIG
# ============================================================
def get_config(mode, model_type):
    if mode == "fast":
        return dict(
            realnvp=dict(layers=6, hidden=128, depth=2),
            maf=dict(layers=4, hidden=128, blocks=2),
            nsf=dict(layers=4, hidden=128, blocks=2, bins=8),
            epochs=500,
            batch_size=256,
        )

    elif mode == "accurate":
        return dict(
            realnvp=dict(layers=10, hidden=256, depth=3),
            maf=dict(layers=8, hidden=256, blocks=4),
            nsf=dict(layers=6, hidden=128, blocks=2, bins=8),
            epochs=1200,
            batch_size=256,
        )

    else:
        raise ValueError("mode must be 'fast' or 'accurate'")


# ============================================================
# BATCHED EVALUATION (IMPORTANT)
# ============================================================
def eval_log_prob(flow, grid, batch_size=1024):
    outs = []
    for i in range(0, grid.shape[0], batch_size):
        batch = grid[i:i + batch_size]
        outs.append(flow.log_prob(batch))
    return torch.cat(outs, dim=0)


# ============================================================
# MAIN FUNCTION
# ============================================================
def run_flow(X_np, grid_np, model_type="nsf", mode="fast"):

    # ----------------------------
    # DEVICE
    # ----------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ----------------------------
    # DATA
    # ----------------------------
    X = torch.from_numpy(np.asarray(X_np, dtype=np.float32)).to(device)
    grid = torch.from_numpy(np.asarray(grid_np, dtype=np.float32)).to(device)

    X = torch.nan_to_num(X)
    grid = torch.nan_to_num(grid)

    # ----------------------------
    # NORMALIZATION
    # ----------------------------
    mean = X.mean(0)
    std = X.std(0) + 1e-6

    X = (X - mean) / std
    grid = (grid - mean) / std

    # small noise (helps generalization)
    X = X + 0.01 * torch.randn_like(X)

    # ----------------------------
    # CONFIG
    # ----------------------------
    cfg = get_config(mode, model_type)

    # ----------------------------
    # MODEL
    # ----------------------------
    if model_type == "realnvp":
        flow = build_realnvp(dim=2, **cfg["realnvp"])
    elif model_type == "maf":
        flow = build_maf(dim=2, **cfg["maf"])
    elif model_type == "nsf":
        flow = build_nsf(dim=2, **cfg["nsf"])
    else:
        raise ValueError("Unknown model type")

    flow = flow.float().to(device)

    # ----------------------------
    # TRAINING
    # ----------------------------
    optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    flow.train()

    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(X.shape[0])

        for i in range(0, X.shape[0], cfg["batch_size"]):
            batch = X[perm[i:i + cfg["batch_size"]]]

            optimizer.zero_grad()
            loss = -flow.log_prob(batch).mean()

            if torch.isnan(loss):
                raise ValueError("NaN loss encountered")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()

        scheduler.step()

    # ----------------------------
    # EVALUATION (batched!)
    # ----------------------------
    flow.eval()
    with torch.no_grad():
        log_p = eval_log_prob(flow, grid, batch_size=1024)
        log_p = log_p.to("cpu", dtype=torch.float32)
        p = torch.exp(log_p).numpy()

    return p