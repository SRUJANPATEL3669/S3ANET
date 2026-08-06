import torch
import torch.nn as nn

class SaliencyMaskGenerator:
    """
    Computes a Saliency Map using the gradients of the model's loss with respect to 
    the input image to identify the most important original spectral bands.
    Generates a binary mask selecting the top N most important bands.
    """
    def __init__(self, model, top_N, criterion=None):
        self.model = model
        self.top_N = top_N
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

    def extract_mask(self, x, y):
        x_clone = x.clone().detach()
        x_clone.requires_grad_(True)
        
        self.model.zero_grad()
        outputs = self.model(x_clone)
        
        loss = self.criterion(outputs, y)
        loss.backward()
        
        # Absolute gradient magnitude as saliency
        saliency = x_clone.grad.data.abs()
        
        # Average saliency scores across batch and spatial dimensions
        w = saliency.mean(dim=(0, 2, 3))
        
        c = w.shape[0] # Total number of input bands
        top_N = min(self.top_N, c)
        
        # Sort w in descending order and select top N band indices
        _, top_n_indices = torch.topk(w, top_N)
        
        batch_size = x.size(0)
        h, w_spatial = x.size(2), x.size(3)
        
        # Create a binary mask M matching the input shape
        mask = torch.zeros((batch_size, c, h, w_spatial), device=x.device)
        mask[:, top_n_indices, :, :] = 1.0
        
        return mask

def Masked_Linf_FGSM(model, images, labels, mask, epsilon, criterion=None, min_val=0.0, max_val=1.0):
    """
    Masked Fast Gradient Sign Method (L-infinity constrained).
    Noise is generated only on the bands selected by the binary mask and clamped by epsilon.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    images_adv = images.clone().detach()
    images_adv.requires_grad = True

    model.zero_grad()
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    loss.backward()

    # Step: alpha = epsilon, using sign of gradient
    g_raw = images_adv.grad.data
    g = mask * g_raw
    delta = epsilon * g.sign()
    
    # Apply noise and clamp to image bounds
    images_adv = torch.clamp(images + delta, min_val, max_val).detach()
    
    return images_adv


def Masked_Linf_IFGSM(model, images, labels, mask, epsilon, alpha, iters, criterion=None, min_val=0.0, max_val=1.0):
    """
    Masked Iterative Fast Gradient Sign Method (L-infinity constrained).
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    images_adv = images.clone().detach()

    for i in range(iters):
        images_adv.requires_grad = True
        
        model.zero_grad()
        outputs = model(images_adv)
        loss = criterion(outputs, labels)
        loss.backward()
        
        g_raw = images_adv.grad.data
        g = mask * g_raw
        
        # Apply step
        images_adv = images_adv.detach() + alpha * g.sign()
        
        # L-infinity Projection
        eta = torch.clamp(images_adv - images, min=-epsilon, max=epsilon)
        images_adv = torch.clamp(images + eta, min_val, max_val).detach()
        
    return images_adv


def Masked_Linf_PGD(model, images, labels, mask, epsilon, alpha, iters, criterion=None, min_val=0.0, max_val=1.0):
    """
    Masked Projected Gradient Descent (L-infinity constrained).
    Includes random initialization strictly on the masked bands.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    # Random initialization within epsilon bound, strictly on masked bands
    noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
    noise = mask * noise
    
    images_adv = images.clone().detach() + noise
    images_adv = torch.clamp(images_adv, min_val, max_val).detach()

    for i in range(iters):
        images_adv.requires_grad = True
        
        model.zero_grad()
        outputs = model(images_adv)
        loss = criterion(outputs, labels)
        loss.backward()
        
        g_raw = images_adv.grad.data
        g = mask * g_raw
        
        # Apply step
        images_adv = images_adv.detach() + alpha * g.sign()
        
        # L-infinity Projection
        eta = torch.clamp(images_adv - images, min=-epsilon, max=epsilon)
        images_adv = torch.clamp(images + eta, min_val, max_val).detach()
        
    return images_adv
