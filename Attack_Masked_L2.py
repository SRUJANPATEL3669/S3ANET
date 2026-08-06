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
        """
        Runs a backward pass to compute gradients on the clean image x.
        
        Args:
            x: Input tensor (e.g., clean image) to run through the model.
            y: Ground truth labels for computing the loss.
        Returns:
            mask: Binary mask of shape (batch_size, c, h, w) where c is the input channel dimension.
        """
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


def Masked_L2_FGSM(model, images, labels, mask, epsilon, criterion, min_val=0.0, max_val=1.0):
    """
    Masked L2-FGSM (Single-Step Attack)
    """
    images_adv = images.clone().detach()
    images_adv.requires_grad_(True)
    
    outputs = model(images_adv)
    loss = criterion(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    g_raw = images_adv.grad.data
    g = mask * g_raw
    
    # Calculate L2 norm of the gradient independently for each sample in the batch
    batch_size = images.size(0)
    g_norm = g.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1)
    
    # Scale the perturbation directly to the full L2 budget epsilon
    delta = epsilon * (g / (g_norm + 1e-10))
    
    images_adv = torch.clamp(images + delta, min_val, max_val).detach()
    return images_adv


def Masked_L2_IFGSM(model, images, labels, mask, epsilon, num_iter=10, alpha=None, criterion=None, min_val=0.0, max_val=1.0):
    """
    Masked L2-I-FGSM (Iterative FGSM / Basic Iterative Method)
    """
    if alpha is None:
        alpha = 2.5 * epsilon / num_iter
        
    x_adv = images.clone().detach()
    batch_size = images.size(0)
    
    for _ in range(num_iter):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        g_raw = x_adv.grad.data
        g = mask * g_raw
        
        # Calculate L2 norm independently per batch
        g_norm = g.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1)
        x_adv = x_adv + alpha * (g / (g_norm + 1e-10))
        
        # Calculate total perturbation and L2 Project back
        delta = x_adv - images
        delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1)
        
        factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-10))
        delta = delta * factor
        
        x_adv = torch.clamp(images + delta, min_val, max_val).detach()
        
    return x_adv


def Masked_L2_PGD(model, images, labels, mask, epsilon, num_iter=10, alpha=None, criterion=None, min_val=0.0, max_val=1.0):
    """
    Masked L2-PGD (Iterative with Random Start)
    """
    if alpha is None:
        alpha = 2.5 * epsilon / num_iter
        
    batch_size = images.size(0)
    
    # Initialize with random uniform noise strictly within the targeted bands
    noise = mask * torch.empty_like(images).uniform_(-epsilon, epsilon)
    noise_norm = noise.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1)
    
    # Project initial noise to be within the L2 ball epsilon
    factor = torch.min(torch.ones_like(noise_norm), epsilon / (noise_norm + 1e-10))
    noise = noise * factor
    
    x_adv = torch.clamp(images + noise, min_val, max_val).detach()
    
    for _ in range(num_iter):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        g_raw = x_adv.grad.data
        g = mask * g_raw
        
        # Step sizing
        g_norm = g.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1)
        x_adv = x_adv + alpha * (g / (g_norm + 1e-10))
        
        # Calculate total perturbation and L2 Project back
        delta = x_adv - images
        delta_norm = delta.view(batch_size, -1).norm(p=2, dim=1).view(batch_size, 1, 1, 1)
        
        factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-10))
        delta = delta * factor
        
        x_adv = torch.clamp(images + delta, min_val, max_val).detach()
        
    return x_adv
