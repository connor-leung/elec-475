import torch
import torch.nn.functional as F


def response_kd_loss(z_s, z_t, y, alpha=0.5, beta=0.5, tau=4.0, ignore_index=255):
    """Response-based KD loss.
    z_s, z_t: (N, C, H, W) student and teacher logits. y: (N, H, W) labels."""
    # Hard term: cross-entropy with the ground truth (T = 1)
    first = F.cross_entropy(z_s, y, ignore_index=ignore_index)

    # Soft term: cross-entropy between the teacher's and student's probabilities, both softened by tau
    p_t = F.softmax(z_t / tau, dim=1)
    log_p_s = F.log_softmax(z_s / tau, dim=1)
    soft = -(p_t * log_p_s).sum(dim=1)  # (N, H, W): one value per pixel
    second = soft[y != ignore_index].mean()

    return alpha * first + beta * second


def feature_kd_loss(f_s, f_t):
    """Feature-based KD loss: 1 - cosine similarity, averaged over all positions.
    f_s: (N, C, h, w) projected student features. f_t: (N, C, H, W) teacher features."""
    # Match the teacher's spatial size (e.g. the student's stride-32 high feature vs the teacher's stride-8 layer4)
    f_s = F.interpolate(f_s, size=f_t.shape[-2:], mode='bilinear', align_corners=False)
    # Cosine similarity across channels at every pixel: (N, H, W)
    cos = F.cosine_similarity(f_s, f_t, dim=1)
    return (1 - cos).mean()
