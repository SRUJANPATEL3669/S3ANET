"""Train MambaHSI (single run, no seed-averaging loop) on PaviaU and save a
checkpoint.

Unlike SACNet/S3ANet, the upstream repo's own train_MambaHSI.py DOES
checkpoint via torch.save — but it loops over 10 random seeds by default,
which is wasteful when all we need is one real trained classifier to attack.
This mirrors its exact data pipeline, model config, and training procedure
for a single seed.

Requires the real `mamba-ssm` package (compiled CUDA selective-scan kernels)
— see the RTAA project notes for the install procedure (needs
--no-build-isolation and a working nvcc >= 11.6; took ~7 min to compile here).

Data: symlink SAFER's PaviaU.mat/PaviaU_gt.mat into
/home/jayant/projects/MambaHSI/data/UP/ (matches upstream's `load_data('UP', ...)`
expected layout) rather than downloading.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
from torchvision import transforms

MAMBAHSI_REPO_DIR = "/home/jayant/projects/MambaHSI"
sys.path.insert(0, MAMBAHSI_REPO_DIR)
from model.MambaHSI import MambaHSI
from utils import data_load_operate
from utils.HSICommonUtils import ImageStretching
from utils.Loss import head_loss


def train(
    n_epochs: int = 200,
    train_samples: int = 30,
    val_samples: int = 10,
    lr: float = 3e-4,
    hidden_dim: int = 128,
    seed: int = 0,
    device: str = "cuda:0",
) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    data, gt = data_load_operate.load_data("UP", f"{MAMBAHSI_REPO_DIR}/data")
    height, width, channels = data.shape
    gt_reshape = gt.reshape(-1)
    class_count = int(np.max(np.unique(gt)))

    img = ImageStretching(data)
    x = transforms.ToTensor()(np.array(img)).unsqueeze(0).float().to(device)

    train_idx, val_idx, test_idx, _ = data_load_operate.sampling(
        [0.1, 0.01], [train_samples, val_samples], gt_reshape, class_count, 1
    )
    train_label, val_label, test_label = data_load_operate.generate_image_iter(
        data, height, width, gt_reshape, (train_idx, val_idx, test_idx)
    )
    train_label, val_label, test_label = train_label.to(device), val_label.to(device), test_label.to(device)

    model = MambaHSI(in_channels=channels, num_classes=class_count, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc, best_state = -1.0, None
    tic = time.time()
    for epoch in range(n_epochs):
        model.train()
        y_train = train_label.unsqueeze(0)
        logits = model(x)
        loss = head_loss(loss_fn, logits, y_train.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x)
            seg_logits = torch.nn.functional.interpolate(
                val_logits, size=val_label.shape, mode="bilinear", align_corners=True
            )
            pred = seg_logits.argmax(1).squeeze(0)
            mask = val_label != -1
            val_acc = (pred[mask] == val_label[mask]).float().mean().item()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1}/{n_epochs} loss={loss.item():.4f} val_acc={val_acc:.4f}")
    train_time = time.time() - tic

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(x)
        seg_logits = torch.nn.functional.interpolate(
            test_logits, size=test_label.shape, mode="bilinear", align_corners=True
        )
        pred = seg_logits.argmax(1).squeeze(0)
        mask = test_label != -1
        test_acc = (pred[mask] == test_label[mask]).float().mean().item()

    return model, {
        "OA": test_acc, "best_val_acc": best_val_acc, "n_epochs": n_epochs,
        "train_time_sec": train_time, "n_classes": class_count, "n_bands": channels,
    }


if __name__ == "__main__":
    model, metadata = train()
    torch.save(model.state_dict(), "checkpoints/mambahsi_paviau.pt")
    print(f"test OA={metadata['OA']*100:.3f}")
    print("saved checkpoints/mambahsi_paviau.pt")
