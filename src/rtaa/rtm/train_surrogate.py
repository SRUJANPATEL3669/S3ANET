"""Training loop for `RTMSurrogate` against `RTMSimulationData`.

Usage once real MODTRAN data exists (schema in simulation_io.py):
    uv run python -m rtaa.rtm.train_surrogate --data path/to/sim_data.npz --out checkpoint.pt

Today, with no simulation data yet, run against the placeholder physics model
to validate the pipeline (`--placeholder`); see placeholder_physics.py for why
that data is not scientifically meaningful.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from rtaa.rtm.placeholder_physics import generate_placeholder_dataset
from rtaa.rtm.simulation_io import RTMSimulationData
from rtaa.rtm.surrogate import RTMSurrogate


class RTMSimulationDataset(Dataset):
    def __init__(self, data: RTMSimulationData):
        self.atm_state = torch.from_numpy(data.atm_state)
        self.transmittance = torch.from_numpy(data.transmittance)
        self.path_radiance = torch.from_numpy(data.path_radiance)

    def __len__(self) -> int:
        return self.atm_state.shape[0]

    def __getitem__(self, idx: int):
        return self.atm_state[idx], self.transmittance[idx], self.path_radiance[idx]


def train(
    data: RTMSimulationData,
    n_epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[RTMSurrogate, dict[str, list[float]]]:
    data.validate()
    torch.manual_seed(seed)

    dataset = RTMSimulationDataset(data)
    n_val = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = RTMSurrogate(n_bands=data.n_bands).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _epoch in range(n_epochs):
        model.train()
        train_losses = []
        for atm_state, t_atm_true, l_path_true in train_loader:
            atm_state = atm_state.to(device)
            t_atm_true, l_path_true = t_atm_true.to(device), l_path_true.to(device)

            t_atm_pred, l_path_pred = model(atm_state)
            loss = mse(t_atm_pred, t_atm_true) + mse(l_path_pred, l_path_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for atm_state, t_atm_true, l_path_true in val_loader:
                atm_state = atm_state.to(device)
                t_atm_true, l_path_true = t_atm_true.to(device), l_path_true.to(device)
                t_atm_pred, l_path_pred = model(atm_state)
                loss = mse(t_atm_pred, t_atm_true) + mse(l_path_pred, l_path_true)
                val_losses.append(loss.item())

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))

    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="Path to .npz matching simulation_io.RTMSimulationData schema")
    parser.add_argument("--placeholder", action="store_true", help="Use analytic placeholder physics instead of --data")
    parser.add_argument("--n-bands", type=int, default=103, help="Only used with --placeholder")
    parser.add_argument("--n-samples", type=int, default=4000, help="Only used with --placeholder")
    parser.add_argument("--out", type=str, default="rtm_surrogate.pt")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.placeholder:
        source_label = "placeholder_physics"
        wavelengths_nm = np.linspace(400, 2500, args.n_bands).astype(np.float32)
        atm_state, transmittance, path_radiance = generate_placeholder_dataset(
            wavelengths_nm, n_samples=args.n_samples
        )
        data = RTMSimulationData(atm_state, transmittance, path_radiance, wavelengths_nm)
    elif args.data:
        source_label = args.data
        data = RTMSimulationData.load(args.data)
    else:
        raise SystemExit("Must pass either --data <path.npz> or --placeholder")

    model, history = train(data, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
    torch.save(model.state_dict(), args.out)

    metadata = {
        "source": source_label,
        "is_real_rtm_data": source_label != "placeholder_physics",
        "n_bands": data.n_bands,
        "n_samples": data.n_samples,
        "n_epochs": args.n_epochs,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = Path(args.out).with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved checkpoint to {args.out}")
    print(f"Saved metadata to {meta_path}")
    print(f"Final train_loss={history['train_loss'][-1]:.6f} val_loss={history['val_loss'][-1]:.6f}")
    if not metadata["is_real_rtm_data"]:
        print("WARNING: trained on analytic placeholder physics, not real MODTRAN output. "
              "Not scientifically usable — retrain once real simulation data is available.")


if __name__ == "__main__":
    main()
