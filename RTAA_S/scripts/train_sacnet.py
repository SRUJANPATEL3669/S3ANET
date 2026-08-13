"""Train SACNet on a specified dataset.

External dependencies (HyperTools, Models) are imported lazily inside
train() after sys.path is patched with SACNET_REPO_DIR so this module can be
imported without the SACNet repo present (e.g. for type-checking locally).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add src to path so rtaa package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from rtaa.models.sacnet import SACNET_REPO_DIR  # noqa: E402


def train(
    dataset: str,
    n_classes: int,
    n_bands: int,
    n_epochs: int = 1000,
    lr: float = 5e-4,
    weight_decay: float = 5e-5,
    device: str = "cuda:0",
) -> tuple:
    # Lazily import repo modules after path is configured
    if SACNET_REPO_DIR not in sys.path:
        sys.path.insert(0, SACNET_REPO_DIR)
    from HyperTools import CalAccuracy  # type: ignore
    from Models import CrossEntropy2d, SACNet, adjust_learning_rate  # type: ignore

    data_dir = f"{SACNET_REPO_DIR}/Data/{dataset}/"
    X = np.load(data_dir + "X.npy")
    _, h, w = X.shape
    Y = np.load(data_dir + "Y.npy")
    train_array = np.load(data_dir + "train_array.npy")
    test_array = np.load(data_dir + "test_array.npy")

    Y_train = np.full(Y.shape, 255)
    Y_train[train_array] = Y[train_array]

    _device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = SACNet(num_features=n_bands, num_classes=n_classes).to(_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    images = torch.from_numpy(X.reshape(1, n_bands, h, w)).float().to(_device)
    labels = torch.from_numpy(Y_train.reshape(1, h, w)).long().to(_device)
    criterion = CrossEntropy2d().to(_device)

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
    return model, {
        "OA": OA,
        "kappa": kappa,
        "n_epochs": n_epochs,
        "train_time_sec": train_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--bands", type=int, required=True)
    args = parser.parse_args()
    _model, _meta = train(args.dataset, args.classes, args.bands)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(_model.state_dict(), f"checkpoints/sacnet_{args.dataset.lower()}.pt")
    print(f"OA={_meta['OA']*100:.3f} Kappa={_meta['kappa']*100:.3f}")
