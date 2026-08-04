import os
import time
import argparse
import torch
import pandas as pd
from torch.autograd import Variable
from HyperTools import *
from Model_S3ANet import *
import logging
import utils_logger

DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'Houston', 4: 'IndianP'}

def run_fgsm(dataID, args):
    # Fix all random seeds for full reproducibility
    import random
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if dataID == 1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './Data/PaviaU/'
    elif dataID == 2:
        num_classes = 16
        num_features = 204
        save_pre_dir = './Data/Salinas/'
    elif dataID == 3:
        num_classes = 7
        num_features = 48           # Houston13 has 48 spectral bands
        save_pre_dir = './Data/Houston/'
    elif dataID == 4:
        num_classes = 16
        num_features = 200
        save_pre_dir = './Data/IndianP/'

    X = np.load(save_pre_dir + 'X.npy')
    _, h, w = X.shape
    Y = np.load(save_pre_dir + 'Y.npy')

    X_train = np.reshape(X, (1, num_features, h, w))
    train_array = np.load(save_pre_dir + 'train_array.npy')
    test_array  = np.load(save_pre_dir + 'test_array.npy')

    Y_train = np.ones(Y.shape) * 255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train, (1, h, w))

    Y_tar = np.zeros(Y.shape, dtype=np.int64).reshape(1, h, w)

    save_path_prefix = args.save_path_prefix + 'Exp_' + DataName[dataID] + '/'
    os.makedirs(save_path_prefix, exist_ok=True)

    # ------ Model ------
    Model = S3ANet(num_features=num_features, num_classes=num_classes, bins=args.bins).cuda()
    Model.train()
    optimizer  = torch.optim.Adam(Model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion  = CrossEntropy2d().cuda()
    images     = torch.from_numpy(X_train).float().cuda()
    label      = torch.from_numpy(Y_train).long().cuda()

    # ------ Train ------
    tr1 = time.time()
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, args.lr, epoch, args.epoch)
        optimizer.zero_grad()
        output   = Model(images)
        seg_loss = criterion(output, label)
        seg_loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print('  epoch %d/%d  loss=%.4f' % (epoch+1, args.epoch, seg_loss.item()))
    tr_time = time.time() - tr1

    Model.eval()

    # ------ Clean predictions ------
    with torch.no_grad():
        clean_out = Model(images)
    _, clean_labels = torch.max(clean_out, 1)
    clean_labels = np.squeeze(clean_labels.cpu().numpy()).reshape(-1)

    # ------ FGSM Attack ------
    processed_image = Variable(images, requires_grad=True)
    label_tar = torch.from_numpy(Y_tar).long().cuda()

    te1 = time.time()
    output   = Model(processed_image)
    seg_loss = criterion(output, label_tar)
    seg_loss.backward()

    adv_noise = args.epsilon * processed_image.grad.data / \
                torch.norm(processed_image.grad.data, float("inf"))
    adv_image_tensor = torch.clamp(processed_image.data - adv_noise, 0, 1)

    X_adv = adv_image_tensor.cpu().numpy()[0]              # (C, H, W)
    X_adv_4d = X_adv.reshape(1, num_features, h, w)

    with torch.no_grad():
        adv_out = Model(torch.from_numpy(X_adv_4d).float().cuda())
    _, adv_labels = torch.max(adv_out, 1)
    adv_labels = np.squeeze(adv_labels.cpu().numpy()).reshape(-1)
    te_time = time.time() - te1

    # ------ Standard Metrics ------
    OA, kappa, ProducerA = CalAccuracy(adv_labels[test_array], Y[test_array])
    AA = np.mean(ProducerA)

    # ------ Advanced Metrics (test pixels only, C-last format) ------
    X_flat_clean = X_train.reshape(num_features, -1).T   # (N, C)
    X_flat_adv   = X_adv.reshape(num_features, -1).T     # (N, C)
    Xc_test = X_flat_clean[test_array]
    Xa_test = X_flat_adv[test_array]

    sam  = CalSAM(Xc_test, Xa_test)
    sid  = CalSID(Xc_test, Xa_test)
    asr  = CalASR(clean_labels, adv_labels, Y, test_array)
    # Limit to 2000 pixels for unmixing speed
    phys = CalPhysConsistency(Xc_test[:2000], Xa_test[:2000], theta=0.1)

    # ------ Save result image ------
    img = DrawResult(np.reshape(adv_labels + 1, -1), dataID)
    plt.imsave(save_path_prefix + 'S3ANet_FGSM_OA' + repr(int(OA*10000)) +
               '_eps' + str(args.epsilon) + '.png', img)

    print('--------- %s ---------' % DataName[dataID])
    print('OA=%.2f%%  Kappa=%.4f  AA=%.2f%%' % (OA*100, kappa, AA*100))
    print('SAM=%.4f rad  SID=%.4f  ASR=%.2f%%  PhysCons=%.2f%%'
          % (sam, sid, asr*100, phys*100))
    print('Train_time=%.1fs  Attack_time=%.1fs' % (tr_time, te_time))

    return dict(Dataset=DataName[dataID],
                OA_pct=round(OA*100, 3), Kappa=round(kappa, 4),
                AA_pct=round(AA*100, 3), Epsilon=args.epsilon,
                SAM_rad=round(sam, 4), SID=round(sid, 4),
                ASR_pct=round(asr*100, 3),
                PhysConsistency_pct=round(phys*100, 3),
                Train_time_s=round(tr_time, 1),
                Attack_time_s=round(te_time, 1))


def main(args):
    dataset_ids = list(range(1, 5)) if args.dataID == 0 else [args.dataID]
    results = []

    for did in dataset_ids:
        print('\n' + '='*55)
        print('  FGSM on Dataset: %s' % DataName[did])
        print('='*55)
        try:
            row = run_fgsm(did, args)
            results.append(row)
        except FileNotFoundError as e:
            print('  [SKIP] Data not found: %s' % str(e))
        except Exception as e:
            print('  [ERROR] %s: %s' % (DataName[did], str(e)))

    if results:
        df = pd.DataFrame(results)
        excel_path = args.save_path_prefix + 'FGSM_All_Metrics_Results.xlsx'
        df.to_excel(excel_path, index=False)
        print('\nResults saved to:', excel_path)
        print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataID=0 means run ALL datasets
    parser.add_argument('--dataID', type=int, default=0,
                        help='1=PaviaU 2=Salinas 3=Houston 4=IndianP  0=ALL')
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--lr',      type=float, default=5e-4)
    parser.add_argument('--decay',   type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=0.04)
    parser.add_argument('--epoch',   type=int,   default=1000)
    parser.add_argument('--bins', nargs='+', type=int, default=[1,2,3,6])

    args = parser.parse_args()
    main(args)
