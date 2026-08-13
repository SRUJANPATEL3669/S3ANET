"""Train HybridSN (Tier 1) to convergence on a real HSI benchmark.

This is what an RTAA attack should actually be evaluated against — attacking
a randomly-initialized classifier produces a near-flat, physically
meaningless loss landscape (confirmed while debugging: gradients through the
full RTAA pipeline are correctly nonzero and finite even then, but an
untrained classifier's decision surface is too close to constant for the
attack numbers to mean anything).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rtaa.data.hsi_dataset import HSIPatchDataset, load_hsi_cube
from rtaa.models.hybridsn import HybridSN


def stratified_train_test_split(labels: list[int], train_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    train_idx, test_idx = [], []
    for cls in np.unique(labels_arr):
        cls_idx = np.nonzero(labels_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_train = max(1, int(len(cls_idx) * train_fraction))
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())
    return train_idx, test_idx


def train(
    dataset_name: str = "IndianPines",
    patch_size: int = 25,
    pca_components: int = 30,
    train_fraction: float = 0.7,
    batch_size: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda:0",
    seed: int = 0,
) -> tuple[HybridSN, dict]:
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    cube, labels = load_hsi_cube(dataset_name)
    ds = HSIPatchDataset(cube, labels, patch_size=patch_size, pca_components=pca_components)

    entry_labels = [entry.label for entry in ds.index]
    train_idx, test_idx = stratified_train_test_split(entry_labels, train_fraction, seed)
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)

    model = HybridSN(
        n_bands=cube.shape[-1], n_classes=ds.n_classes, patch_size=patch_size, pca_components=pca_components
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(n_epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_correct += (logits.argmax(dim=1) == y).sum().item()
                test_total += x.size(0)
        test_acc = test_correct / test_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch+1}/{n_epochs}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return model, {
        "dataset": dataset_name,
        "n_classes": ds.n_classes,
        "n_bands": cube.shape[-1],
        "patch_size": patch_size,
        "pca_components": pca_components,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
        "n_epochs": n_epochs,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="IndianPines")
    parser.add_argument("--patch-size", type=int, default=25)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out", type=str, default="checkpoints/hybridsn_indianpines.pt")
    args = parser.parse_args()

    model, metadata = train(
        dataset_name=args.dataset,
        patch_size=args.patch_size,
        pca_components=args.pca_components,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved checkpoint to {out_path}")
    print(f"Final test accuracy: {metadata['final_test_acc']:.4f}")


if __name__ == "__main__":
    main()
