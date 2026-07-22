import os
import time
import argparse
import random
import torch
from torch.autograd import Variable
from HyperTools import *
from Model_S3ANet import *
import logging
import utils_logger

DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'Houston',4:'IndianP'}

def set_seed(seed=42):
    """Fix all random seeds for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if args.dataID == 1:
        num_classes = 9
        num_features = 103
        save_pre_dir = './Data/PaviaU/'
    elif args.dataID == 2:
        num_classes = 16
        num_features = 204
        save_pre_dir = './Data/Salinas/'
    elif args.dataID == 3:
        num_classes = 15
        num_features = 144
        save_pre_dir = './Data/Houston/'
    elif args.dataID == 4:
        num_classes = 16
        num_features = 200
        save_pre_dir = './Data/IndianP/'

    X = np.load(save_pre_dir + 'X.npy')
    num_features, h, w = X.shape
    Y = np.load(save_pre_dir + 'Y.npy')
    num_classes = int(Y.max()) + 1

    X_train = np.reshape(X, (1, 1, num_features, h, w))  # 5D: [batch, depth, channels, H, W]
    train_array = np.load(save_pre_dir + 'train_array.npy')
    test_array = np.load(save_pre_dir + 'test_array.npy')
    Y_train = np.ones(Y.shape) * 255
    Y_train[train_array] = Y[train_array]
    Y_train = np.reshape(Y_train, (1, h, w))

    # define the targeted label in the attack
    Y_tar = np.zeros(Y.shape)
    Y_tar = np.reshape(Y_tar, (1, h, w))

    save_path_prefix = args.save_path_prefix + 'Exp_' + DataName[args.dataID] + '/'
    save_log_prefix = args.save_path_prefix + 'log_' + DataName[args.dataID] + '/'  # save_log_path
    log_path = save_log_prefix + args.model + '.log'


    if os.path.exists(save_path_prefix) == False:
        os.makedirs(save_path_prefix)
    if os.path.exists(save_log_prefix) == False:
        os.makedirs(save_log_prefix)

    if args.model == 'S3ANet':
        Model = S3ANet(num_features=num_features, num_classes=num_classes, bins=args.bins).to(device)
        num_epochs = args.epoch

        Model.train()
        optimizer = torch.optim.Adam(Model.parameters(), lr=args.lr,weight_decay=args.decay)


        images_5d = torch.from_numpy(X_train).float().to(device)  # [batch, depth, channels, H, W]
        b, d, c, h_dim, w_dim = images_5d.shape
        images = images_5d.view(b * d, c, h_dim, w_dim)  # reshape to 4D for Conv2d
        label = torch.from_numpy(Y_train).long().to(device)
        criterion = CrossEntropy2d().to(device)

        t1 = time.time()
        # train the classification model

        # Train time #
        tr1_time = time.time()
        for epoch in range(num_epochs):
            adjust_learning_rate(optimizer, args.lr, epoch, args.epoch)
            tem_time = time.time()
            optimizer.zero_grad()
            output = Model(images)

            seg_loss = criterion(output,label)
            seg_loss.backward()

            optimizer.step()
            # scheduler.step()

            batch_time = time.time() - tem_time
            if (epoch + 1) % 1 == 0:
                print('epoch %d/%d:  time: %.2f cls_loss = %.3f' % (epoch + 1, num_epochs, batch_time, seg_loss.item()))
        tr2_time = time.time()-tr1_time

        Model.eval()

        # ── Clean predictions ────────────────────────────────────────────────
        with torch.no_grad():
            clean_output = Model(images)
        _, clean_labels = torch.max(clean_output, 1)
        clean_pred = np.squeeze(clean_labels.cpu().numpy()).reshape(-1)

        # Save clean classification map image (only generated once; skipped if already exists)
        OA_clean, kappa_clean, ProducerA_clean = CalAccuracy(clean_pred[test_array], Y[test_array])
        AA_clean = np.mean(ProducerA_clean)
        clean_map_path = (save_path_prefix + args.model + '_clean'
                          + '_OA'    + repr(int(OA_clean    * 10000))
                          + '_kappa' + repr(int(kappa_clean * 10000))
                          + '.png')
        if not os.path.exists(clean_map_path):
            img_clean = DrawResult(np.reshape(clean_pred + 1, -1), args.dataID, h, w)
            plt.imsave(clean_map_path, img_clean)
            print('Clean map saved -> ' + clean_map_path)
        else:
            print('Clean map already exists, skipping -> ' + clean_map_path)

        # Keep original numpy image for SAM / SID / physical-consistency
        X_orig_np = images.cpu().data.numpy()[0]          # (C, H, W)

        # adversarial attack
        processed_image = Variable(images)
        processed_image = processed_image.requires_grad_()
        label_tar = torch.from_numpy(Y_tar).long().to(device)

        # 生成对抗样本
        output  = Model(processed_image)
        seg_loss = criterion(output, label_tar)
        #### Test time #####
        te1_time = time.time()
        seg_loss.backward()
        adv_noise = args.epsilon * processed_image.grad.data / torch.norm(processed_image.grad.data, float("inf"))

        processed_image.data = processed_image.data - adv_noise

        X_adv_4d = torch.clamp(processed_image, 0, 1).cpu().data.numpy()  # [batch*depth, channels, H, W]
        X_adv_4d = np.reshape(X_adv_4d, (b, d, num_features, h, w))       # restore to 5D: [batch, depth, channels, H, W]
        X_adv = X_adv_4d[0, 0]                                              # [channels, H, W] for saving
        X_adv = np.reshape(X_adv, (1, 1, num_features, h, w))               # 5D adv image for model input

        adv_images_5d = torch.from_numpy(X_adv).float().to(device)  # [batch, depth, channels, H, W]
        b2, d2, c2, h2, w2 = adv_images_5d.shape
        adv_images = adv_images_5d.view(b2 * d2, c2, h2, w2)  # reshape to 4D for Conv2d

        # 对抗样本用于测试
        output = Model(adv_images)
        _, predict_labels = torch.max(output, 1)

        te2_time = time.time() - te1_time

        predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)
        # results on the adversarial test set
        OA2, kappa2, ProducerA2 = CalAccuracy(predict_labels[test_array], Y[test_array])
        AA2 = np.mean(ProducerA2)

        img = DrawResult(np.reshape(predict_labels + 1, -1), args.dataID, h, w)
        plt.imsave(save_path_prefix + args.model + '_FGSM_OA' + repr(int(OA2 * 10000)) + '_kappa' + repr(
            int(kappa2 * 10000)) + 'Epsilon' + str(args.epsilon) + '.png', img)

        # ── Spectral / physical attack-quality metrics ───────────────────────
        # X_adv_np: adversarial image as (C, H, W) numpy array
        X_adv_np  = torch.clamp(processed_image, 0, 1).detach().cpu().numpy()[0]  # (C,H,W)

        sam_val   = CalSAM(X_orig_np, X_adv_np)
        sid_val   = CalSID(X_orig_np, X_adv_np)
        print('Computing physical-consistency rate (may take a moment)...')
        phys_rate = CalPhysicalConsistency(X_orig_np, X_adv_np, theta=5.0)
        asr_val   = CalASR(clean_pred, predict_labels, Y, test_array)

        ######
        print('─' * 55)
        print('── Clean Baseline ───────────────────────────────────')
        print('OA    : %.3f %%' % (OA_clean    * 100))
        print('Kappa : %.3f %%' % (kappa_clean * 100))
        print('AA    : %.3f %%' % (AA_clean    * 100))
        print('─' * 55)
        print('── After FGSM Attack (ε=%.4f) ───────────────────────' % args.epsilon)
        print('OA    : %.3f %%' % (OA2    * 100))
        print('Kappa : %.3f %%' % (kappa2 * 100))
        print('AA    : %.3f %%' % (AA2    * 100))
        print('producerA:', (ProducerA2) * 100)
        print('Train_time: %.2f, Test_time: %.2f, Runtime: %.2f' % (tr2_time, te2_time, tr2_time + te2_time))
        print('─' * 55)
        print('── Spectral Attack Metrics ──────────────────────────')
        print('SAM  (mean spectral angle, deg)  : %.4f' % sam_val)
        print('SID  (spectral info divergence)  : %.6f' % sid_val)
        print('Physical-consistency rate (θ=5°) : %.4f  (%.2f%%)' % (phys_rate, phys_rate * 100))
        print('ASR  (attack success rate)        : %.4f  (%.2f%%)' % (asr_val,   asr_val   * 100))
        print('─' * 55)
        # ── Notebook-compatible summary (parsed by S3ANet_Experiments.ipynb) ──
        print('OA=%.3f,Kappa=%.3f' % (OA2 * 100, kappa2 * 100))
        print('AA=%.3f' % (AA2 * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataID', type=int, default=1)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='S3ANet')

    # train
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=0.04)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--bins', nargs='+',type=int, default=[1, 2, 3, 6])

    args = parser.parse_args()
    main(args)
