"""Train SACNet on a specified dataset."""
import sys, time, argparse, numpy as np, torch
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from rtaa.models.sacnet import SACNET_REPO_DIR
sys.path.insert(0, SACNET_REPO_DIR)
from HyperTools import CalAccuracy  # type: ignore
from Models import CrossEntropy2d, SACNet, adjust_learning_rate  # type: ignore

def train(dataset, n_classes, n_bands, n_epochs=1000, lr=5e-4, weight_decay=5e-5, device="cuda:0"):
    data_dir = f"{SACNET_REPO_DIR}/Data/{dataset}/"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--bands", type=int, required=True)
    args = parser.parse_args()
    model, metadata = train(args.dataset, args.classes, args.bands)
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/sacnet_{args.dataset.lower()}.pt")
    print(f"OA={metadata['OA']*100:.3f} Kappa={metadata['kappa']*100:.3f}")
