import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  
import torch

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm    
        
def DrawResult(labels,imageID,h=None,w=None):
    #ID=1:Pavia University
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        palette = np.array([[216,191,216],
                            [0,255,0],
                            [0,255,255],
                            [45,138,86],
                            [255,0,255],
                            [255,165,0],
                            [159,31,239],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    
    elif imageID ==2:
        row = 512
        col = 217
        palette = np.array([[37, 58, 150],
                            [47, 78, 161],
                            [56, 87, 166],
                            [56, 116, 186],
                            [51, 181, 232],
                            [112, 204, 216],
                            [119, 201, 168],
                            [148, 204, 120],
                            [188, 215, 78],
                            [238, 234, 63],
                            [246, 187, 31],
                            [244, 127, 33],
                            [239, 71, 34],
                            [238, 33, 35],
                            [180, 31, 35],
                            [123, 18, 20]])
        palette = palette*1.0/255

    elif imageID == 3:
        row = 349
        col = 1905
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255],
                            [216, 191, 216],
                            [255, 0, 0],
                            [139, 0, 0],
                            [0, 0, 0],
                            [255, 255, 0],
                            [238, 154, 0],
                            [85, 26, 139],
                            [255, 127, 80]])
        palette = palette * 1.0 / 255

    elif imageID == 4:
        row = 145
        col = 145
        palette = np.array([[255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [176, 48, 96],
                            [46, 139, 87],
                            [160, 32, 240],
                            [255, 127, 80],
                            [127, 255, 212],
                            [218, 112, 214],
                            [160, 82, 45],
                            [127, 255, 0],
                            [216, 191, 216],
                            [238, 0, 0]])
        palette = palette * 1.0 / 255

    elif imageID == 5:
        row = 550
        col = 400
        palette = np.array([[255, 0, 0],
                            [239, 155, 0],
                            [255, 255, 0],
                            [0, 255, 0],
                            [0, 255, 255],
                            [0, 140, 140],
                            [0, 0, 255],
                            [255, 255, 255],
                            [160, 32, 240]])
        palette = palette * 1.0 / 255
    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    if h is not None and w is not None:
        row = h
        col = w
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    max_lbl = int(np.max(label)) if label.size > 0 else 0
    correct_sum = np.zeros((max_lbl+1))
    reali = np.zeros((max_lbl+1))
    predicti = np.zeros((max_lbl+1))
    producerA = np.zeros((max_lbl+1))
    
    for i in range(0, max_lbl+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        if reali[i] > 0:
            producerA[i] = correct_sum[i] / reali[i]
   
    denom = (n*n - np.sum(reali * predicti))
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / denom if denom != 0 else 0.0
    return OA,Kappa,producerA

def LoadHSI(dataID=1,num_label=150):
    #ID=1:Pavia University
    if dataID==1:        
        data = sio.loadmat('./Data/PaviaU.mat')
        X = data['paviaU']    
        data = sio.loadmat('./Data/PaviaU_gt.mat')
        Y = data['paviaU_gt']
            
    elif dataID==2:        
        data = sio.loadmat('./Data/Salinas_corrected.mat')
        X = data['salinas_corrected']    
        data = sio.loadmat('./Data/Salinas_gt.mat')
        Y = data['salinas_gt']

    elif dataID==3:
        import os
        if os.path.exists('./Data/GRSS2013.mat'):
            data = sio.loadmat('./Data/GRSS2013.mat')
            X = data['GRSS2013']
            data = sio.loadmat('./Data/GRSS2013_gt.mat')
            Y = data['GRSS2013_gt']
        else:
            try:
                data = sio.loadmat('./Data/Houston13.mat')
                X = data.get('Houston13', data.get('houston', data.get('GRSS2013')))
                data_gt = sio.loadmat('./Data/Houston13_7gt.mat')
                Y = data_gt.get('Houston13_7gt', data_gt.get('houston_gt', data_gt.get('GRSS2013_gt', data_gt.get('Houston13_gt'))))
            except NotImplementedError:
                import h5py
                with h5py.File('./Data/Houston13.mat', 'r') as f:
                    valid_keys = [k for k in f.keys() if not k.startswith('#')]
                    X = np.array(f[valid_keys[0]]).T
                with h5py.File('./Data/Houston13_7gt.mat', 'r') as f:
                    valid_keys_gt = [k for k in f.keys() if not k.startswith('#')]
                    Y = np.array(f[valid_keys_gt[0]]).T
    elif dataID==4:
        data = sio.loadmat('./Data/Indian_pines_corrected.mat')
        X = data['indian_pines_corrected']
        data = sio.loadmat('./Data/Indian_pines_gt.mat')
        Y = data['indian_pines_gt']
        num_label = [30, 50, 50, 50, 50, 50, 20, 50, 15, 50, 50, 50, 50, 30, 50, 50]


    [row,col,n_feature] = X.shape
    K = row*col
    X = X.reshape(K, n_feature)       
    
    n_class = int(Y.max())

    X = featureNormalize(X,2)  
    X = np.reshape(X,(row,col,n_feature))
    X = np.moveaxis(X,-1,0)
    Y = Y.reshape(K,).astype('int')


    for i in range(1,n_class+1):
        
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        np.random.seed(12345)
        randomArray_label = np.random.permutation(n_data)
        if isinstance(num_label, list):
            train_num = num_label[i-1]
        else:
            train_num = num_label
        if i==1:
            train_array = index[randomArray_label[0:train_num]]
            test_array = index[randomArray_label[train_num:n_data]]
        else:            
            train_array = np.append(train_array,index[randomArray_label[0:train_num]])
            test_array = np.append(test_array,index[randomArray_label[train_num:n_data]])

    return X,Y,train_array,test_array


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Attack Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────

def CalSAM(X_orig, X_adv):
    """
    Spectral Angle Mapper (SAM) — physical-distance metric.

    Measures the mean spectral angle (in degrees) between original and
    adversarial pixels.  A larger angle means the perturbation moves pixels
    further from their original spectral signature.

    Parameters
    ----------
    X_orig : ndarray, shape (C, H, W) or (N, C)
        Original (clean) hyperspectral image / pixel array.
    X_adv  : ndarray, same shape as X_orig
        Adversarial hyperspectral image / pixel array.

    Returns
    -------
    mean_sam : float
        Mean SAM across all pixels, in degrees.
    """
    # Flatten to (N, C)
    if X_orig.ndim == 3:
        C, H, W = X_orig.shape
        X_orig = X_orig.reshape(C, -1).T          # (N, C)
        X_adv  = X_adv.reshape(C, -1).T

    eps = 1e-10
    dot   = np.sum(X_orig * X_adv, axis=1)
    norm1 = np.linalg.norm(X_orig, axis=1) + eps
    norm2 = np.linalg.norm(X_adv,  axis=1) + eps
    cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    sam_per_pixel = np.degrees(np.arccos(cos_angle))   # radians → degrees
    mean_sam = float(np.mean(sam_per_pixel))
    return mean_sam


def CalSID(X_orig, X_adv):
    """
    Spectral Information Divergence (SID) — material-identity metric.

    SID(p, q) = D_KL(p||q) + D_KL(q||p), where p and q are probability
    distributions derived from spectral vectors by L1-normalisation.
    A larger SID implies more change in spectral material identity.

    Parameters
    ----------
    X_orig : ndarray, shape (C, H, W) or (N, C)
    X_adv  : ndarray, same shape as X_orig

    Returns
    -------
    mean_sid : float
        Mean SID across all pixels (dimensionless).
    """
    if X_orig.ndim == 3:
        C, H, W = X_orig.shape
        X_orig = X_orig.reshape(C, -1).T
        X_adv  = X_adv.reshape(C, -1).T

    eps = 1e-10
    # Shift to non-negative before L1-normalising (data should already be ≥0)
    p = np.abs(X_orig) + eps
    q = np.abs(X_adv)  + eps
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)

    kl_pq = np.sum(p * np.log(p / q + eps), axis=1)
    kl_qp = np.sum(q * np.log(q / p + eps), axis=1)
    sid_per_pixel = kl_pq + kl_qp
    mean_sid = float(np.mean(sid_per_pixel))
    return mean_sid


def CalPhysicalConsistency(X_orig, X_adv, theta=5.0, n_endmembers=None):
    """
    Physical-consistency rate — fraction of adversarial pixels that pass an
    unmixing round-trip test.

    Algorithm
    ---------
    1. Estimate endmembers from X_orig via VCA-lite (random vertex search).
    2. Solve NNLS abundances for every pixel in X_adv.
    3. Reconstruct each pixel as A @ endmembers.
    4. Compute SAM between X_adv pixel and its reconstruction.
    5. Rate = fraction of pixels with SAM < theta (degrees).

    Parameters
    ----------
    X_orig      : ndarray, shape (C, H, W) or (N, C)  — clean image
    X_adv       : ndarray, same shape                  — adversarial image
    theta       : float  (default 5.0°)                — SAM threshold
    n_endmembers: int or None — number of endmembers;
                  defaults to min(C, 10) if None.

    Returns
    -------
    rate : float  ∈ [0, 1]   fraction of physically consistent pixels
    """
    from scipy.optimize import nnls

    # Flatten to (N, C)
    if X_orig.ndim == 3:
        C, H, W = X_orig.shape
        orig_2d = X_orig.reshape(C, -1).T    # (N, C)
        adv_2d  = X_adv.reshape(C, -1).T
    else:
        orig_2d = X_orig.copy()
        adv_2d  = X_adv.copy()

    N, C = orig_2d.shape
    if n_endmembers is None:
        n_endmembers = min(C, 10)

    # ── Endmember extraction (simplified VCA-style random search) ─────────────
    # Save global numpy random state so the local seed(0) here does not
    # corrupt the reproducibility guarantee established by set_seed(42).
    _rng_state = np.random.get_state()
    np.random.seed(0)
    endmember_idx = [np.random.randint(N)]
    for _ in range(n_endmembers - 1):
        # pick the pixel furthest from the current endmember set (greedy)
        E = orig_2d[endmember_idx]               # (k, C)
        dists = np.min(
            np.sum((orig_2d[:, None, :] - E[None, :, :]) ** 2, axis=2),
            axis=1
        )
        endmember_idx.append(int(np.argmax(dists)))
    endmembers = orig_2d[endmember_idx]           # (n_endmembers, C)
    np.random.set_state(_rng_state)               # restore global state

    # ── Abundance estimation + reconstruction ─────────────────────────────────
    E_T = endmembers.T                             # (C, n_endmembers)
    eps = 1e-10
    consistent = 0
    for pixel in adv_2d:
        abund, _ = nnls(E_T, pixel)
        recon    = E_T @ abund                     # (C,)
        # SAM between adversarial pixel and its reconstruction
        cos_a = np.dot(pixel, recon) / (
            np.linalg.norm(pixel) * np.linalg.norm(recon) + eps
        )
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if angle < theta:
            consistent += 1

    rate = consistent / N
    return rate


def CalASR(clean_pred, adv_pred, Y_true, test_array):
    """
    Attack Success Rate (ASR) — fraction of correctly-classified test pixels
    that are flipped to a wrong class by the adversarial attack.

    ASR = |{i ∈ test : clean_pred[i] == Y_true[i]  AND
                        adv_pred[i]   != Y_true[i]}|
          ─────────────────────────────────────────────
                |{i ∈ test : clean_pred[i] == Y_true[i]}|

    Parameters
    ----------
    clean_pred : ndarray, shape (N,)   — predictions on clean image
    adv_pred   : ndarray, shape (N,)   — predictions on adversarial image
    Y_true     : ndarray, shape (N,)   — ground-truth labels
    test_array : ndarray               — indices of test pixels

    Returns
    -------
    asr : float ∈ [0, 1]
    """
    y_true_test   = Y_true[test_array]
    clean_test    = clean_pred[test_array]
    adv_test      = adv_pred[test_array]

    # Pixels that the clean model got right
    correctly_classified = (clean_test == y_true_test)
    n_correct = int(correctly_classified.sum())
    if n_correct == 0:
        return 0.0

    # Among those, how many did the attack flip?
    flipped = correctly_classified & (adv_test != y_true_test)
    asr = float(flipped.sum()) / n_correct
    return asr


def Apply_S3ANet_Defense(X_adv):
    """
    S3ANet Spatial-Spectral Defense Algorithm:
    Removes high-frequency adversarial perturbations from hyperspectral data
    by applying spatial median filtering combined with spectral smoothing,
    restoring physical spatial-spectral manifold consistency.

    Parameters
    ----------
    X_adv : ndarray, shape (C, H, W) or torch.Tensor (1, C, H, W)

    Returns
    -------
    X_defended : same type and shape as X_adv
    """
    is_tensor = False
    if isinstance(X_adv, torch.Tensor):
        is_tensor = True
        device = X_adv.device
        X_np = X_adv.detach().cpu().numpy()
        was_4d = (X_np.ndim == 4)
        if was_4d:
            X_np = X_np[0]
    else:
        X_np = X_adv.copy()
        was_4d = (X_np.ndim == 4)
        if was_4d:
            X_np = X_np[0]

    from scipy.ndimage import median_filter
    C, H, W = X_np.shape
    X_def_spatial = np.zeros_like(X_np)

    # 1. Spatial median filtering per band to eliminate spatial adversarial noise spikes
    for c in range(C):
        X_def_spatial[c] = median_filter(X_np[c], size=3)

    # 2. Spectral smoothing (3-band moving average) to enforce spectral continuity
    X_def_spectral = np.zeros_like(X_def_spatial)
    for c in range(C):
        c_min = max(0, c - 1)
        c_max = min(C, c + 2)
        X_def_spectral[c] = np.mean(X_def_spatial[c_min:c_max], axis=0)

    # 3. Blend spatial and spectral defense outputs
    X_defended = 0.5 * X_def_spatial + 0.5 * X_def_spectral
    X_defended = np.clip(X_defended, 0.0, 1.0)

    if is_tensor:
        if was_4d:
            X_defended = np.expand_dims(X_defended, axis=0)
        return torch.from_numpy(X_defended).float().to(device)
    else:
        if was_4d:
            X_defended = np.expand_dims(X_defended, axis=0)
        return X_defended