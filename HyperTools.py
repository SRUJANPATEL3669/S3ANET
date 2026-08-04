import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

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
        
def DrawResult(labels,imageID):
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
        row = 210
        col = 954
        palette = np.array([[0, 205, 0],
                            [127, 255, 0],
                            [46, 139, 87],
                            [0, 139, 0],
                            [160, 82, 45],
                            [0, 255, 255],
                            [255, 255, 255]])
        palette = palette * 1.0 / 255
        num_class = min(num_class, len(palette))

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
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))
    
    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    return OA,Kappa,producerA

# ============================================================
#  Advanced Adversarial Metrics
# ============================================================

def CalSAM(X_clean, X_adv):
    """Mean Spectral Angle Mapper (radians) between clean and adversarial pixels.
    X_clean, X_adv: (N, C) arrays.
    """
    dot = np.sum(X_clean * X_adv, axis=1)
    norm1 = np.linalg.norm(X_clean, axis=1)
    norm2 = np.linalg.norm(X_adv, axis=1)
    cos_theta = np.clip(dot / (norm1 * norm2 + 1e-8), -1.0, 1.0)
    return float(np.mean(np.arccos(cos_theta)))

def CalSID(X_clean, X_adv):
    """Mean Spectral Information Divergence between clean and adversarial pixels.
    X_clean, X_adv: (N, C) arrays.
    """
    X_c = X_clean - np.min(X_clean, axis=1, keepdims=True) + 1e-4
    X_a = X_adv  - np.min(X_adv,  axis=1, keepdims=True) + 1e-4
    P = X_c / np.sum(X_c, axis=1, keepdims=True)
    Q = X_a / np.sum(X_a, axis=1, keepdims=True)
    sid = np.sum(P * np.log(P / Q), axis=1) + np.sum(Q * np.log(Q / P), axis=1)
    return float(np.mean(sid))

def CalASR(clean_preds, adv_preds, Y_flat, test_array):
    """Attack Success Rate: fraction of correctly-classified test pixels that got flipped.
    clean_preds, adv_preds, Y_flat: 1-D arrays over ALL pixels.
    test_array: indices of test pixels.
    """
    c = clean_preds[test_array]
    a = adv_preds[test_array]
    t = Y_flat[test_array]
    correct_mask = (c == t)
    if correct_mask.sum() == 0:
        return 0.0
    return float(np.sum(correct_mask & (a != t)) / correct_mask.sum())

def _vca(Y, R):
    """Lightweight Vertex Component Analysis (VCA) to find R endmembers.
    Y: (C, N)  spectral matrix.
    Returns E: (C, R) endmember matrix.
    """
    C, N = Y.shape
    u, _, _ = np.linalg.svd(Y, full_matrices=False)
    Ud = u[:, :R]                          # (C, R)
    x_p = Ud.T @ Y                         # (R, N)
    c   = x_p.mean(axis=1, keepdims=True)  # (R, 1)
    y   = x_p - c                          # (R, N)
    indice = np.zeros(R, dtype=int)
    A = np.zeros((R, R))
    A[-1, 0] = 1.0
    for i in range(R):
        w = np.random.randn(R, 1)
        f = w - A @ (np.linalg.pinv(A) @ w)
        f /= (np.linalg.norm(f) + 1e-8)
        v = f.T @ y                        # (1, N)
        indice[i] = np.argmax(np.abs(v))
        A[:, i] = y[:, indice[i]]
    return Y[:, indice]                    # (C, R)

def CalPhysConsistency(X_clean, X_adv, num_endmembers=10, theta=0.1):
    """Physical-consistency rate: fraction of adversarial pixels that pass an
    unmixing round-trip (re-unmix X_adv, reconstruct, SAM to X_adv < theta).
    X_clean, X_adv: (N, C)
    """
    from scipy.optimize import nnls
    np.random.seed(0)
    Y_c = X_clean.T                        # (C, N)
    R   = num_endmembers

    # Extract endmembers from clean subset (max 5000 pixels for speed)
    sub_n = min(Y_c.shape[1], 5000)
    idx   = np.random.choice(Y_c.shape[1], sub_n, replace=False)
    E     = _vca(Y_c[:, idx], R)          # (C, R)

    Y_adv = X_adv.T                        # (C, N)
    N     = Y_adv.shape[1]
    abundances = np.zeros((R, N))
    delta = 1e-4
    E_aug = np.vstack([E, np.ones((1, R)) * delta])
    for i in range(N):
        b_aug = np.append(Y_adv[:, i], delta)
        abundances[:, i], _ = nnls(E_aug, b_aug)

    X_recon = (E @ abundances).T           # (N, C)
    sam_vals = np.arccos(np.clip(
        np.sum(X_adv * X_recon, axis=1) /
        (np.linalg.norm(X_adv, axis=1) * np.linalg.norm(X_recon, axis=1) + 1e-8),
        -1.0, 1.0))
    return float(np.mean(sam_vals < theta))


def LoadHSI(dataID=1,num_label=150):
    #ID=1:Pavia University
    if dataID==1:        
        data = sio.loadmat('./Data/paviaU.mat')
        X = data['paviaU']    
        data = sio.loadmat('./Data/paviaU_gt.mat')
        Y = data['paviaU_gt']
            
    elif dataID==2:        
        data = sio.loadmat('./Data/salinas_corrected.mat')
        X = data['salinas_corrected']    
        data = sio.loadmat('./Data/salinas_gt.mat')
        Y = data['salinas_gt']

    elif dataID==3:
        # Houston13 is stored in MATLAB v7.3 (HDF5) format.
        # h5py reads in C order: ori_data -> (C=48, dim1=954, dim2=210)
        # Transpose to (row=210, col=954, C=48)
        import h5py
        with h5py.File('./Data/houston2013.mat', 'r') as f:
            raw = np.array(f['ori_data'])          # (48, 954, 210)
            X = raw.transpose(2, 1, 0)             # (210, 954, 48)
        with h5py.File('./Data/houston2013_gt.mat', 'r') as f:
            Y_raw = np.array(f['map'])             # (954, 210) in h5py C-order
            Y = Y_raw.T.astype('int')              # (210, 954)

    elif dataID==4:
        data = sio.loadmat('./Data/indian_pines_corrected.mat')
        X = data['indian_pines_corrected']
        data = sio.loadmat('./Data/indian_pines_gt.mat')
        Y = data['indian_pines_gt']
        num_label = [30, 50, 50, 50, 50, 50, 20, 50, 15, 50, 50, 50, 50, 30, 50, 50]


    [row,col,n_feature] = X.shape
    K = row*col
    X = X.reshape(K, n_feature)       
    
    n_class = Y.max()

    X = featureNormalize(X,2)  
    X = np.reshape(X,(row,col,n_feature))
    X = np.moveaxis(X,-1,0)
    Y = Y.reshape(K,).astype('int')


    for i in range(1,n_class+1):
        
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        np.random.seed(12345)
        randomArray_label = np.random.permutation(n_data)
        # num_label can be a list (per-class counts for Indian Pines) or a scalar
        train_num = int(num_label[i-1]) if isinstance(num_label, list) else int(num_label)
        if i==1:
            train_array = index[randomArray_label[0:train_num]]
            test_array = index[randomArray_label[train_num:n_data]]
        else:            
            train_array = np.append(train_array,index[randomArray_label[0:train_num]])
            test_array = np.append(test_array,index[randomArray_label[train_num:n_data]])

    return X,Y,train_array,test_array