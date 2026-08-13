from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

def overall_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean() * 100.0

def per_class_accuracy(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> list[float]:
    accs = []
    for c in range(n_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            accs.append((preds[mask] == c).mean() * 100.0)
        else:
            accs.append(0.0)
    return accs

def average_accuracy(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    accs = per_class_accuracy(preds, labels, n_classes)
    valid_accs = [a for c, a in zip(range(n_classes), accs) if (labels == c).sum() > 0]
    return np.mean(valid_accs) if valid_accs else 0.0

def cohen_kappa(preds: np.ndarray, labels: np.ndarray) -> float:
    return cohen_kappa_score(labels, preds)

def attack_success_rate(clean_preds: np.ndarray, adv_preds: np.ndarray, labels: np.ndarray) -> float:
    correct_mask = (clean_preds == labels)
    if correct_mask.sum() == 0:
        return 0.0
    flipped = (adv_preds[correct_mask] != labels[correct_mask]).sum()
    return (flipped / correct_mask.sum()) * 100.0

def spectral_angle_mapper(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """Mean SAM in degrees over the spatial/pixel dimensions."""
    # Assuming inputs are (..., n_bands)
    clean = clean.reshape(-1, clean.shape[-1])
    adv = adv.reshape(-1, adv.shape[-1])
    dot = (clean * adv).sum(dim=1)
    norm_c = torch.norm(clean, dim=1)
    norm_a = torch.norm(adv, dim=1)
    cos_theta = torch.clamp(dot / (norm_c * norm_a + 1e-12), -1.0, 1.0)
    angles = torch.acos(cos_theta)
    return float((angles.mean().item() * 180.0 / np.pi))

def spectral_information_divergence(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """Mean SID over the spatial/pixel dimensions."""
    clean = clean.reshape(-1, clean.shape[-1])
    adv = adv.reshape(-1, adv.shape[-1])
    
    p = clean / (clean.sum(dim=1, keepdim=True) + 1e-12)
    q = adv / (adv.sum(dim=1, keepdim=True) + 1e-12)
    
    sid_pq = (p * torch.log(p / (q + 1e-12) + 1e-12)).sum(dim=1)
    sid_qp = (q * torch.log(q / (p + 1e-12) + 1e-12)).sum(dim=1)
    
    return float((sid_pq + sid_qp).mean().item())

def physical_consistency_rate(clean: torch.Tensor, adv: torch.Tensor, threshold_deg: float = 3.0) -> float:
    """
    Physical-consistency rate: fraction of adv pixels that pass an unmixing round-trip.
    Re-unmix X_adv, reconstruct, SAM to X_adv < θ.
    Simplified unmixing using PCA as endmember basis.
    """
    clean_flat = clean.reshape(-1, clean.shape[-1])
    adv_flat = adv.reshape(-1, adv.shape[-1])
    
    # Mean centering
    mean_clean = clean_flat.mean(dim=0, keepdim=True)
    centered = clean_flat - mean_clean
    
    _, _, V = torch.pca_lowrank(centered, q=min(5, clean.shape[-1]))
    
    # Project adv onto V and back
    centered_adv = adv_flat - mean_clean
    abundances = centered_adv @ V
    reconstructed = (abundances @ V.T) + mean_clean
    
    # Calculate SAM between adv and reconstructed
    dot = (adv_flat * reconstructed).sum(dim=1)
    norm_a = torch.norm(adv_flat, dim=1)
    norm_r = torch.norm(reconstructed, dim=1)
    cos_theta = torch.clamp(dot / (norm_a * norm_r + 1e-12), -1.0, 1.0)
    angles = torch.acos(cos_theta) * 180.0 / np.pi
    
    return float((angles < threshold_deg).float().mean().item() * 100.0)
