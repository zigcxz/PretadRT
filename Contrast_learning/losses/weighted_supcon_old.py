import torch
import torch.nn.functional as F


def weighted_supcon_loss(
    features,
    positive_mask,
    positive_weights,
    temperature=0.07
):
    """
    Weighted supervised contrastive loss.

    Parameters
    ----------
    features:
        [N, D]

    positive_mask:
        [N, N] bool

    positive_weights:
        [N, N] float

    temperature:
        contrastive temperature
    """

    if features.ndim != 2:
        raise ValueError(
            "features must have shape [N, D]"
        )

    positive_mask = (
        positive_mask.bool()
    )

    positive_weights = (
        positive_weights.float()
    )

    # ========================================
    # Normalize embedding
    # ========================================

    features = F.normalize(
        features,
        p=2,
        dim=1
    )

    # ========================================
    # Pairwise cosine similarity
    # ========================================

    logits = (
        torch.matmul(
            features,
            features.T
        )
        /
        temperature
    )

    n = features.size(0)

    self_mask = torch.eye(
        n,
        dtype=torch.bool,
        device=features.device
    )

    # ========================================
    # denominator不包括自己
    # ========================================

    denominator_logits = (
        logits.masked_fill(
            self_mask,
            float("-inf")
        )
    )

    log_denominator = (
        torch.logsumexp(
            denominator_logits,
            dim=1,
            keepdim=True
        )
    )

    log_prob = (
        logits
        -
        log_denominator
    )

    # diagonal无意义，置0防止0 * inf
    log_prob = log_prob.masked_fill(
        self_mask,
        0.0
    )

    # ========================================
    # Positive weights
    # ========================================

    positive_mask = (
        positive_mask
        &
        (~self_mask)
    )

    weights = (
        positive_weights
        *
        positive_mask.float()
    )

    positive_weight_sum = (
        weights.sum(dim=1)
    )

    # 有positive的样本才能作为anchor
    valid_anchor = (
        positive_weight_sum > 0
    )

    if not torch.any(
        valid_anchor
    ):

        # 保持computational graph
        return (
            features.sum() * 0.0
        )

    # ========================================
    # Weighted SupCon
    # ========================================

    loss_per_anchor = -(
        weights
        *
        log_prob
    ).sum(dim=1) / (
        positive_weight_sum
        +
        1e-12
    )

    loss = (
        loss_per_anchor[
            valid_anchor
        ]
        .mean()
    )

    return loss