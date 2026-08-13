"""Train SACNet from scratch on PaviaU and save a checkpoint.

The upstream repo (github.com/YonghaoXu/SACNet) doesn't ship pretrained
weights and its own Test_Clean.py never calls torch.save — this script
mirrors its exact training procedure (same data pipeline via GenSample.py,
same hyperparameters) but adds checkpointing.

Run GenSample.py in the SACNet repo first:
    cd /home/jayant/projects/SACNet && python GenSample.py --dataID 1 --train_samples 300
(requires Data/PaviaU.mat + Data/PaviaU_gt.mat — symlink from the SAFER
mirror rather than downloading, see rtaa/data/hsi_dataset.py for that mirror)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from rtaa.models.sacnet import SACNET_REPO_DIR

sys.path.insert(0, SACNET_REPO_DIR)
from HyperTools import CalAccuracy
from Models import CrossEntropy2d, SACNet, adjust_learning_rate


def train(
    n_classes: int = 9,
    n_bands: int = 103,
    n_epochs: int = 1000,
    lr: float = 5e-4,
    weight_decay: float = 5e-5,
    device: str = "cuda:0",
) -> tuple[torch.nn.Module, dict]:
    data_dir = f"{SACNET_REPO_DIR}/Data/PaviaU/"
    X = np.load(data_dir + "X.npy")
    _, h, w = X.shape
    Y = np.load(data_dir + "Y.npy")
    train_array = np.load(data_dir + "train_array.npy")
    test_array = np.load(data_dir + "test_array.npy")

    Y_train = np.full(Y.shape, 255)
    Y_train[train_array] = Y[train_array]

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = SACNet(num_features=n_bands, num_classes=n_classes).to(device)
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

    return model, {"OA": OA, "kappa": kappa, "n_epochs": n_epochs, "train_time_sec": train_time}


if __name__ == "__main__":
    model, metadata = train()
    torch.save(model.state_dict(), "checkpoints/sacnet_paviau.pt")
    print(f"OA={metadata['OA']*100:.3f} Kappa={metadata['kappa']*100:.3f}")
    print("saved checkpoints/sacnet_paviau.pt")
