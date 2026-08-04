import numpy as np
import scipy.optimize as opt

def compute_sam(X_clean, X_adv):
    """Calculate mean Spectral Angle Mapper (SAM) between clean and adv pixels."""
    dot = np.sum(X_clean * X_adv, axis=1)
    norm1 = np.linalg.norm(X_clean, axis=1)
    norm2 = np.linalg.norm(X_adv, axis=1)
    cos_theta = np.clip(dot / (norm1 * norm2 + 1e-8), -1.0, 1.0)
    sam = np.arccos(cos_theta)
    return np.mean(sam)

def compute_sid(X_clean, X_adv):
    """Calculate mean Spectral Information Divergence (SID)."""
    X_c = X_clean - np.min(X_clean, axis=1, keepdims=True) + 1e-4
    X_a = X_adv - np.min(X_adv, axis=1, keepdims=True) + 1e-4
    
    P = X_c / np.sum(X_c, axis=1, keepdims=True)
    Q = X_a / np.sum(X_a, axis=1, keepdims=True)
    
    D_Px_Py = np.sum(P * np.log(P / Q), axis=1)
    D_Py_Px = np.sum(Q * np.log(Q / P), axis=1)
    sid = D_Px_Py + D_Py_Px
    return np.mean(sid)

def vca(Y, R):
    """Vertex Component Analysis for endmember extraction."""
    L, N = Y.shape
    # SVD for dimensionality reduction
    u, s, v = np.linalg.svd(Y, full_matrices=False)
    Ud = u[:, :R]
    
    x_p = np.dot(Ud.T, Y)
    c = np.mean(x_p, axis=1, keepdims=True)
    y = x_p - c
    
    indice = np.zeros(R, dtype=int)
    A = np.zeros((R, R))
    A[-1, 0] = 1
    
    for i in range(R):
        w = np.random.randn(R, 1)
        f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
        f = f / (np.linalg.norm(f) + 1e-8)
        
        v = np.dot(f.T, y)
        indice[i] = np.argmax(np.abs(v))
        A[:, i] = y[:, indice[i]]
        
    return Y[:, indice]

def compute_physical_consistency(X_clean, X_adv, num_endmembers=10, theta=0.1):
    """
    Physical-consistency rate: fraction of adv pixels that pass an unmixing round-trip.
    Re-unmix X_adv, reconstruct, SAM to X_adv < theta.
    """
    Y = X_clean.T # (C, N)
    if Y.shape[1] > 5000:
        idx = np.random.choice(Y.shape[1], 5000, replace=False)
        Y_sub = Y[:, idx]
    else:
        Y_sub = Y
        
    endmembers = vca(Y_sub, num_endmembers) # (C, R)
    Y_adv = X_adv.T # (C, N)
    
    N = Y_adv.shape[1]
    abundances = np.zeros((num_endmembers, N))
    
    delta = 1e-4
    A_aug = np.vstack([endmembers, np.ones((1, num_endmembers)) * delta])
    
    for i in range(N):
        b_aug = np.append(Y_adv[:, i], delta)
        abundances[:, i], _ = opt.nnls(A_aug, b_aug)
        
    X_reconstructed = np.dot(endmembers, abundances).T # (N, C)
    
    dot = np.sum(X_adv * X_reconstructed, axis=1)
    norm1 = np.linalg.norm(X_adv, axis=1)
    norm2 = np.linalg.norm(X_reconstructed, axis=1)
    cos_theta = np.clip(dot / (norm1 * norm2 + 1e-8), -1.0, 1.0)
    sam_vals = np.arccos(cos_theta)
    
    pass_rate = np.mean(sam_vals < theta)
    return pass_rate

def compute_asr(clean_preds, adv_preds, labels, test_array):
    """
    ASR: fraction of correctly-classified test pixels flipped.
    """
    clean_test_preds = clean_preds[test_array]
    adv_test_preds = adv_preds[test_array]
    true_labels = labels[test_array]
    
    correct_idx = (clean_test_preds == true_labels)
    flipped_idx = (adv_test_preds != true_labels)
    
    if np.sum(correct_idx) == 0:
        return 0.0
        
    asr = np.sum(correct_idx & flipped_idx) / np.sum(correct_idx)
    return asr
