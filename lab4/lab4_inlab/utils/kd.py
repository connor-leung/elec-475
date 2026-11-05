import torch
import torch.nn.functional as F


def response_kd_loss(z_s, z_t, y, alpha=0.5, beta=0.5, tau=4.0, ignore_index=255):
    """Response-based KD loss.
    z_s, z_t: (N, C, H, W) student and teacher logits. y: (N, H, W) labels."""
    # TODO: complete in Step 3
    raise NotImplementedError


def feature_kd_loss(f_s, f_t):
    """Feature-based KD loss: 1 - cosine similarity, averaged over all positions.
    f_s: (N, C, h, w) projected student features. f_t: (N, C, H, W) teacher features."""
    # TODO: complete in Step 4
    raise NotImplementedError
