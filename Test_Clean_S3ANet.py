import os
import time
import argparse
import random
import torch
import numpy as np
from HyperTools import *
from Model_S3ANet import *

# ─────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────
DataName = {
    1: 'PaviaU',
    2: 'Salinas',
    3: 'Houston',
    4: 'IndianP',
}

DATASET_CFG = {
    1: dict(num_classes=9,  num_features=103, save_pre_dir='./Data/PaviaU/'),
    2: dict(num_classes=16, num_features=204, save_pre_dir='./Data/Salinas/'),
    3: dict(num_classes=15, num_features=144, save_pre_dir='./Data/Houston/'),
    4: dict(num_classes=16, num_features=200, save_pre_dir='./Data/IndianP/'),
}

# ─────────────────────────────────────────────
# Reproducibility helper
# ─────────────────────────────────────────────
def set_seed(seed: int = 42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Main training + clean evaluation loop
# ─────────────────────────────────────────────
def main(args):
    set_seed(42)

    cfg = DATASET_CFG[args.dataID]
    num_features = cfg['num_features']
    save_pre_dir = cfg['save_pre_dir']

    # ── Load pre-processed data ──────────────────────────────────────────────
    X = np.load(save_pre_dir + 'X.npy')                     # (C, H, W)
    num_features, h, w = X.shape
    Y = np.load(save_pre_dir + 'Y.npy')                     # (H*W,)
    num_classes = int(Y.max()) + 1

    train_array = np.load(save_pre_dir + 'train_array.npy')
    test_array  = np.load(save_pre_dir + 'test_array.npy')

    # Build labelled training map (unlabelled pixels → 255)
    Y_train = np.ones(Y.shape) * 255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train, (1, h, w))                # (1, H, W)

    # S3ANet expects a 5-D tensor [batch, depth, C, H, W]; depth=1 here
    X_5d = np.reshape(X, (1, 1, num_features, h, w))

    # ── Output directory ────────────────────────────────────────────────────
    save_path_prefix = args.save_path_prefix + 'Exp_' + DataName[args.dataID] + '/'
    os.makedirs(save_path_prefix, exist_ok=True)

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Model ───────────────────────────────────────────────────────────────
    Model = S3ANet(
        num_features=num_features,
        num_classes=num_classes,
        bins=args.bins
    ).to(device)
    Model.train()

    optimizer = torch.optim.Adam(
        Model.parameters(), lr=args.lr, weight_decay=args.decay
    )

    # Collapse 5-D → 4-D: [batch*depth, C, H, W] for Conv2d layers
    images_5d = torch.from_numpy(X_5d).float().to(device)
    b, d, c, h_dim, w_dim = images_5d.shape
    images = images_5d.view(b * d, c, h_dim, w_dim)

    label     = torch.from_numpy(Y_train).long().to(device)
    criterion = CrossEntropy2d().to(device)

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs = args.epoch
    print(f'\n=== Training S3ANet on {DataName[args.dataID]} for {num_epochs} epochs ===')
    train_start = time.time()

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, num_epochs)

        tem_time = time.time()
        optimizer.zero_grad()

        output   = Model(images)
        seg_loss = criterion(output, label)
        seg_loss.backward()
        optimizer.step()

        batch_time = time.time() - tem_time
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print('epoch %d/%d  time: %.2fs  cls_loss = %.4f'
                  % (epoch + 1, num_epochs, batch_time, seg_loss.item()))

    train_time = time.time() - train_start
    print(f'Training finished in {train_time:.2f}s\n')

    # ── Clean evaluation ─────────────────────────────────────────────────────
    Model.eval()
    with torch.no_grad():
        output = Model(images)

    _, predict_labels = torch.max(output, 1)
    predict_labels = np.squeeze(predict_labels.cpu().numpy()).reshape(-1)

    OA, kappa, ProducerA = CalAccuracy(predict_labels[test_array], Y[test_array])
    AA = np.mean(ProducerA)

    # Save classification map
    img = DrawResult(np.reshape(predict_labels + 1, -1), args.dataID, h, w)
    map_path = (save_path_prefix
                + 'S3ANet_clean'
                + '_OA'    + repr(int(OA    * 10000))
                + '_kappa' + repr(int(kappa * 10000))
                + '.png')
    plt.imsave(map_path, img)

    # ── Print summary ────────────────────────────────────────────────────────
    print('─' * 50)
    print(f'Dataset : {DataName[args.dataID]}')
    print(f'OA      : {OA    * 100:.3f} %')
    print(f'Kappa   : {kappa * 100:.3f} %')
    print(f'AA      : {AA    * 100:.3f} %')
    print('ProducerA (per class):')
    for i, pa in enumerate(ProducerA):
        print(f'  Class {i:2d}: {pa * 100:.3f} %')
    print(f'Map saved -> {map_path}')

    # ── Spectral / physical metrics (clean baseline) ─────────────────────────
    # Get the clean image as (C, H, W) numpy array
    X_clean_np = images.cpu().data.numpy()[0]          # (C, H, W)

    # SAM and SID are 0 by definition (no perturbation); shown for reference
    sam_clean  = 0.0
    sid_clean  = 0.0

    # Physical-consistency rate on the clean image: meaningful baseline value
    # showing how well unperturbed pixels satisfy the unmixing round-trip.
    print('Computing physical-consistency rate on clean image (may take a moment)...')
    phys_clean = CalPhysicalConsistency(X_clean_np, X_clean_np, theta=5.0)

    # ASR = 0 for clean evaluation (no adversarial attack applied)
    asr_clean  = 0.0

    print('── Spectral Metrics (clean baseline) ────────────────────────')
    print('SAM  (mean spectral angle, deg)  : %.4f  [0 = no perturbation]' % sam_clean)
    print('SID  (spectral info divergence)  : %.6f  [0 = no perturbation]' % sid_clean)
    print('Physical-consistency rate (θ=5°) : %.4f  (%.2f%%)' % (phys_clean, phys_clean * 100))
    print('ASR  (attack success rate)        : %.4f  [0 = no attack]'       % asr_clean)
    print('─' * 50)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean-set evaluation of S3ANet on hyperspectral datasets.'
    )
    parser.add_argument('--dataID', type=int, default=0,
                        help='0 = all datasets, 1=PaviaU, 2=Salinas, 3=Houston, 4=IndianP')
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--lr',    type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epoch', type=int,   default=1000)
    parser.add_argument('--bins', nargs='+', type=int, default=[1, 2, 3, 6])

    args = parser.parse_args()

    # Support dataID=0 to iterate over all datasets
    if args.dataID == 0:
        for data_id in DataName.keys():
            print(f'\n{"="*60}')
            print(f'=== Dataset ID {data_id}: {DataName[data_id]} ===')
            print(f'{"="*60}')
            args.dataID = data_id
            main(args)
    else:
        main(args)
