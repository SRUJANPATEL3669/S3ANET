"""Digital pixel-level baseline attacks (H_2 §1.B.1), operating directly on
classifier input patches (post-PCA), for comparison against RTAA."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


def fgsm_attack(
    model: nn.Module,
    patches: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
) -> Tensor:
    patches = patches.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(patches), labels)
    grad = torch.autograd.grad(loss, patches)[0]
    return (patches + epsilon * grad.sign()).detach()


def pgd_attack(
    model: nn.Module,
    patches: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    step_size: float = 0.01,
    n_steps: int = 20,
    random_start: bool = True,
) -> Tensor:
    original = patches.clone().detach()
    if random_start:
        delta = torch.empty_like(original).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(original)

    for _ in range(n_steps):
        adv = (original + delta).detach().requires_grad_(True)
        loss = nn.functional.cross_entropy(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        delta = delta + step_size * grad.sign()
        delta = torch.clamp(delta, -epsilon, epsilon)

    return (original + delta).detach()


def ifgsm_attack(
    model: nn.Module,
    patches: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    n_steps: int = 20,
) -> Tensor:
    """I-FGSM attack (BIM). Like PGD but without random restart, and step size is epsilon / n_steps."""
    return pgd_attack(model, patches, labels, epsilon, epsilon / n_steps, n_steps, random_start=False)


def _spectral_band_clusters(band_variance: Tensor, k: int, generator: torch.Generator) -> Tensor:
    """Ports the band-grouping heuristic from SS-FGSM's official km.py
    (github.com/AAAA-CS/SS_FGSM_HyperspectralAdversarialAttack): bands are
    clustered by similarity of their variance ("covariance" in the original,
    which for a single band reduces to variance) rather than by spectral
    shape, with cluster centers re-estimated as the mean band index of their
    members, iterated 10 times exactly as in the original `kmeans`."""
    n_bands = band_variance.shape[0]
    cen_band = torch.randperm(n_bands, generator=generator)[:k].tolist()
    for _ in range(11):
        dis = (band_variance.unsqueeze(1) - band_variance[cen_band].unsqueeze(0)).abs()  # (n_bands, k)
        assignment = dis.argmin(dim=1)
        new_cen = []
        for c in range(k):
            members = (assignment == c).nonzero(as_tuple=True)[0]
            new_cen.append(int(members.float().mean().round().item()) if len(members) else cen_band[c])
        cen_band = new_cen
    dis = (band_variance.unsqueeze(1) - band_variance[cen_band].unsqueeze(0)).abs()
    assignment = dis.argmin(dim=1)
    return torch.tensor([cen_band[c] for c in assignment.tolist()])


def _smooth_spectral_noise(noise: Tensor, cluster_idx: Tensor) -> Tensor:
    """Ports SS-FGSM's `kmeans_noise` (km.py): for adjacent bands assigned to
    the same cluster, copies the perturbation of the earlier band forward."""
    smoothed = noise.clone()
    for i in range(1, cluster_idx.shape[0]):
        if cluster_idx[i] == cluster_idx[i - 1]:
            smoothed[:, i] = smoothed[:, i - 1]
    return smoothed


def ssfgsm_attack(
    model: nn.Module,
    patches: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    n_steps: int = 20,
    n_band_clusters: int = 20,
    seed: int = 0,
) -> Tensor:
    """SS-FGSM baseline (Yang et al., "spatial-spectral" FGSM), ported from
    the official code at
    github.com/AAAA-CS/SS_FGSM_HyperspectralAdversarialAttack. Iterative
    sign-gradient attack (like BIM) where the perturbation is smoothed before
    each gradient step to promote local coherence.

    The official method smooths in two ways: spatially, by averaging the
    perturbation within SLIC superpixels computed over a full scene; and
    spectrally, by copying the perturbation across bands grouped by variance
    similarity. Only the spectral half has a counterpart here — RTAA's
    evaluation pipeline (see scripts/run_asr_sweep.py) samples individual
    pixel spectra with no spatial neighborhood, so there is no scene to run
    SLIC over. This ports the spectral-smoothing half exactly and omits the
    spatial half.

    Supports arbitrary input shapes — the spectral axis is identified
    automatically and the spectral-clustering smoothing is applied along it.
    """
    original = patches.clone().detach()

    # For non-2D inputs, fall back to PGD (spectral clustering makes little
    # sense on PCA-reduced or multi-dim patches).
    if original.dim() > 3 or (original.dim() == 3 and original.shape[1] == 1):
        return pgd_attack(model, patches, labels, epsilon,
                          epsilon / max(n_steps, 1), n_steps, random_start=False)

    spectra = original.squeeze(-1) if original.dim() == 3 else original
    n_bands = spectra.shape[1]
    k = min(n_band_clusters, n_bands)

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        band_variance = spectra.var(dim=0).detach().cpu()
    cluster_idx = _spectral_band_clusters(band_variance, k, generator).to(spectra.device)

    noise = torch.empty_like(spectra).uniform_(-epsilon, epsilon)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(n_steps):
        noise = _smooth_spectral_noise(noise, cluster_idx).detach().requires_grad_(True)
        adv = spectra + noise
        adv_in = adv.unsqueeze(-1) if original.dim() == 3 else adv
        loss = loss_fn(model(adv_in), labels)
        grad = torch.autograd.grad(loss, noise)[0]
        noise = torch.clamp(noise + epsilon * grad.sign(), -epsilon, epsilon).detach()

    adv = torch.clamp(spectra + noise, 0.0, 1.0).detach()
    return adv.unsqueeze(-1) if original.dim() == 3 else adv


def ssfgsm_attack_full_scene(
    model: nn.Module,
    scene: Tensor,
    labels: Tensor,
    epsilon: float = 0.05,
    n_steps: int = 20,
    n_band_clusters: int = 20,
    n_superpixels: int = 100,
    compactness: float = 10.0,
    seed: int = 0,
    ignore_label: int = 255,
) -> Tensor:
    """SS-FGSM for whole-scene models (SACNet/S3ANet), including both spatial and spectral smoothing."""
    try:
        from skimage.segmentation import slic
    except ImportError:
        raise ImportError("skimage is required for full-scene SS-FGSM. pip install scikit-image")

    original = scene.clone().detach()  # (1, n_bands, H, W)
    _, n_bands, h, w = original.shape
    
    # 1. Spectral clustering
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        band_variance = original[0].view(n_bands, -1).var(dim=1).cpu()
    k = min(n_band_clusters, n_bands)
    cluster_idx = _spectral_band_clusters(band_variance, k, generator).to(scene.device)

    # 2. Spatial superpixels
    with torch.no_grad():
        img_np = original[0].mean(dim=0).cpu().numpy()
        segments = slic(img_np.astype(np.float64), n_segments=n_superpixels, compactness=compactness, start_label=0)
        segments_t = torch.from_numpy(segments).to(scene.device).long()
        n_segs = segments_t.max().item() + 1
        
    noise = torch.empty_like(original).uniform_(-epsilon, epsilon)

    def _masked_cross_entropy(logits, lbls):
        mask = lbls != ignore_label
        logits_flat = logits.permute(0, 2, 3, 1)[mask.view(1, h, w)].view(-1, logits.shape[1])
        lbls_flat = lbls[mask]
        return nn.functional.cross_entropy(logits_flat, lbls_flat)

    for _ in range(n_steps):
        # Spectral smooth
        noise_spec = noise.clone()
        for i in range(1, cluster_idx.shape[0]):
            if cluster_idx[i] == cluster_idx[i - 1]:
                noise_spec[:, i] = noise_spec[:, i - 1]
        
        # Spatial smooth (average noise within each superpixel)
        noise_smooth = noise_spec.clone()
        for seg_id in range(n_segs):
            mask = (segments_t == seg_id)  # (H, W)
            if mask.sum() > 0:
                # Average noise over all pixels in this superpixel, per band
                mean_noise = noise_spec[:, :, mask].mean(dim=2, keepdim=True)  # (1, n_bands, 1)
                noise_smooth[:, :, mask] = mean_noise.expand(-1, -1, mask.sum())
                
        noise = noise_smooth.detach().requires_grad_(True)
        adv = original + noise
        logits = model(adv)
        
        loss = _masked_cross_entropy(logits, labels if labels.dim() == 2 else labels.unsqueeze(0))
        grad = torch.autograd.grad(loss, noise)[0]
        noise = torch.clamp(noise + epsilon * grad.sign(), -epsilon, epsilon).detach()

    adv = torch.clamp(original + noise, 0.0, 1.0).detach()
    return adv


def cw_attack(
    model: nn.Module,
    patches: Tensor,
    labels: Tensor,
    n_steps: int = 100,
    lr: float = 0.01,
    c: float = 1.0,
    kappa: float = 0.0,
) -> Tensor:
    """Carlini-Wagner L2 attack (untargeted), f6 formulation. Box-constrains
    adv to [0, 1] (the reflectance domain used throughout this codebase, not
    the [-1, 1] range the vanilla tanh reparametrization gives by default)."""
    original = patches.clone().detach()
    w = torch.atanh(torch.clamp(2 * original - 1, -0.999, 0.999)).detach().requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)

    n_classes = model(original[:1]).shape[-1]
    one_hot = nn.functional.one_hot(labels, n_classes).float()

    for _ in range(n_steps):
        adv = 0.5 * (torch.tanh(w) + 1)
        logits = model(adv)

        real = (logits * one_hot).sum(dim=1)
        other = (logits - one_hot * 1e4).max(dim=1).values
        f_loss = torch.clamp(real - other + kappa, min=0.0).sum()

        l2_loss = (adv - original).pow(2).flatten(1).sum(dim=1).sum()
        loss = l2_loss + c * f_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (0.5 * (torch.tanh(w) + 1)).detach()
