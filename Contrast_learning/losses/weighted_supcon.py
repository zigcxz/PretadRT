import torch
import torch.nn.functional as F


def weighted_supcon_loss(
    features,
    positive_mask,
    positive_weights,
    anchor_weights=None,
    temperature=0.07,
    eps=1e-12,
):
    """
    Weighted supervised contrastive loss with pair-level positive weights
    and optional scaffold-frequency anchor weights.

    Pair-level:
        positive_weights[i, j]

    Anchor-level:
        anchor_weights[i] = 1 / sqrt(n_scaffold(i))

    For each valid anchor i:
        L_i = - sum_j w_ij * log p_ij / sum_j w_ij

    Final reduction:
        L = sum_i a_i * L_i / sum_i a_i

    Passing anchor_weights=None reproduces the original mean reduction.
    """
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {tuple(features.shape)}")

    n = features.size(0)

    if positive_mask.shape != (n, n):
        raise ValueError(f"positive_mask must have shape {(n, n)}, got {tuple(positive_mask.shape)}")
    if positive_weights.shape != (n, n):
        raise ValueError(f"positive_weights must have shape {(n, n)}, got {tuple(positive_weights.shape)}")

    positive_mask = positive_mask.to(device=features.device, dtype=torch.bool)
    positive_weights = positive_weights.to(device=features.device, dtype=features.dtype)

    if anchor_weights is None:
        anchor_weights = torch.ones(n, device=features.device, dtype=features.dtype)
    else:
        if anchor_weights.ndim != 1 or anchor_weights.numel() != n:
            raise ValueError(f"anchor_weights must have shape [{n}], got {tuple(anchor_weights.shape)}")
        anchor_weights = anchor_weights.to(device=features.device, dtype=features.dtype)

    if not torch.isfinite(features).all():
        raise ValueError("features contains NaN/Inf")
    if not torch.isfinite(positive_weights).all():
        raise ValueError("positive_weights contains NaN/Inf")
    if not torch.isfinite(anchor_weights).all():
        raise ValueError("anchor_weights contains NaN/Inf")
    if torch.any(positive_weights < 0):
        raise ValueError("positive_weights must be non-negative")
    if torch.any(anchor_weights < 0):
        raise ValueError("anchor_weights must be non-negative")

    features = F.normalize(features, p=2, dim=1)
    logits = torch.matmul(features, features.T) / temperature

    self_mask = torch.eye(n, dtype=torch.bool, device=features.device)
    denominator_logits = logits.masked_fill(self_mask, float("-inf"))
    log_denominator = torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    log_prob = logits - log_denominator
    log_prob = log_prob.masked_fill(self_mask, 0.0)

    positive_mask = positive_mask & (~self_mask)
    pair_weights = positive_weights * positive_mask.float()
    positive_weight_sum = pair_weights.sum(dim=1)
    valid_anchor = positive_weight_sum > 0

    if not torch.any(valid_anchor):
        return features.sum() * 0.0

    loss_per_anchor = -(pair_weights * log_prob).sum(dim=1) / (positive_weight_sum + eps)

    valid_losses = loss_per_anchor[valid_anchor]
    valid_anchor_weights = anchor_weights[valid_anchor]
    weight_sum = valid_anchor_weights.sum()

    if weight_sum <= eps:
        return features.sum() * 0.0

    loss = (valid_anchor_weights * valid_losses).sum() / (weight_sum + eps)
    return loss
