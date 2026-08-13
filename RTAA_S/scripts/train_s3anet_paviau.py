"""Train S3ANet from scratch on PaviaU and save a checkpoint.

Same situation as SACNet: the upstream repo (github.com/YichuXu/S3ANet)
combines train+attack in one script and never checkpoints. This mirrors its
exact training procedure (same data pipeline, same hyperparameters:
bins=[1,2,3,6], 1000 epochs) with checkpointing added.

Run GenSample.py in the S3ANet repo first:
    cd /home/jayant/projects/S3ANet && python GenSample.py --dataID 1 --train_samples 300
(requires Data/PaviaU.mat + Data/PaviaU_gt.mat — symlink from the SAFER
mirror rather than downloading)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

S3ANET_REPO_DIR = "/home/jayant/projects/S3ANet"

sys.path.insert(0, S3ANET_REPO_DIR)
from HyperTools import CalAccuracy  # type: ignore
from Model_S3ANet import CrossEntropy2d, S3ANet, adjust_learning_rate  # type: ignore


def train(
    n_classes: int = 9,
    n_bands: int = 103,
    bins: tuple[int, ...] = (1, 2, 3, 6),
    n_epochs: int = 1000,
    lr: float = 5e-4,
    weight_decay: float = 5e-5,
    device: str = "cuda:0",
) -> tuple[torch.nn.Module, dict]:
    data_dir = f"{S3ANET_REPO_DIR}/Data/PaviaU/"
    X = np.load(data_dir + "X.npy")
    _, h, w = X.shape
    Y = np.load(data_dir + "Y.npy")
    train_array = np.load(data_dir + "train_array.npy")
    test_array = np.load(data_dir + "test_array.npy")

    Y_train = np.full(Y.shape, 255)
    Y_train[train_array] = Y[train_array]

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = S3ANet(num_features=n_bands, num_classes=n_classes, bins=list(bins)).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    images = torch.from_numpy(X.reshape(1, n_bands, h, w)).float().to(device)
    labels = torch.from_numpy(Y_train.reshape(1, h, w)).long().to(device)
    criterion = CrossEntropy2d().to(device)

    tic = time.time()
    for epoch in range(n_epochs):
        adjust_learning_rate(optimizer, lr, epoch, n_epochs)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"epoch {epoch+1}/{n_epochs} loss={loss.item():.4f}")
    train_time = time.time() - tic

    model.eval()
    with torch.no_grad():
        predict = model(images).argmax(1).squeeze(0).cpu().numpy().reshape(-1)
    OA, kappa, _ = CalAccuracy(predict[test_array], Y[test_array])

    return model, {"OA": OA, "kappa": kappa, "n_epochs": n_epochs, "train_time_sec": train_time, "bins": list(bins)}


if __name__ == "__main__":
    model, metadata = train()
    torch.save(model.state_dict(), "checkpoints/s3anet_paviau.pt")
    print(f"OA={metadata['OA']*100:.3f} Kappa={metadata['kappa']*100:.3f}")
    print("saved checkpoints/s3anet_paviau.pt")
