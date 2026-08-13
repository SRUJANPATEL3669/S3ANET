"""Cross-architecture transferability matrix: does an RTAA attack crafted
against one classifier degrade a *different* classifier it was never
optimized against (black-box transfer), not just the one it was generated
for (white-box, already measured in the per-architecture ASR sweeps)?

Scoped to SACNet <-> S3ANet only, not the full 5-architecture set, for a
concrete data reason: SACNet and S3ANet's data pipelines (`Data/PaviaU/`
under their respective repo clones) turned out to be numerically identical
— same (103, 610, 340) scene, same [0,1] min-max normalization, same
40076-pixel test_array — so an adversarial scene crafted for one is a valid
input to feed the other with no reformatting or renormalization. MambaHSI's
data pipeline uses a different normalization (2nd-98th-percentile stretch,
not min-max) — feeding a perturbation crafted in SACNet/S3ANet's linear
[0,1] space into MambaHSI would silently compare across two different
mappings of the same physical radiance, which isn't a valid transfer test.
Excluded rather than faked; SpectralFormer/HybridSN (different dataset,
Indian Pines) are excluded for the same reason (no shared scene to
transfer within).

For each of 8 seeds: generate RTAA (mismatch-aware) against SACNet and
against S3ANet independently (same masked test-pixel subset per seed for
both, so the two are evaluated on the same held-out pixels), then evaluate
both resulting adversarial scenes against *both* classifiers — 2x2 matrix,
diagonal = white-box (already measured elsewhere, reproduced here as a
consistency check), off-diagonal = black-box transfer. Evaluated under the
"clear" (generation) atmosphere only — transferability is orthogonal to the
atmospheric-severity axis already explored exhaustively elsewhere.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from run_asr_sweep_sacnet import (
    DATA_DIR,
    GENERATION_ATM,
    N_CLASSES,
    RTM_CHECKPOINT,
    build_eval_labels,
    simulate_and_evaluate_scene,
)
from scipy import stats

from rtaa.attacks.sacnet_attack import PhysicalViabilityWeights, SACNetRTAAAttack
from rtaa.models.s3anet import load_s3anet
from rtaa.models.sacnet import load_sacnet
from rtaa.rtm.mismatch import AtmosphericMismatchConfig
from rtaa.rtm.surrogate import RTMSurrogate

EPSILON = 0.01
N_STEPS = 25
N_SAMPLES = 3000
N_SEEDS = 8
CHECKPOINTS = {"sacnet": "checkpoints/sacnet_paviau.pt", "s3anet": "checkpoints/s3anet_paviau.pt"}


def run_one_seed(seed: int, X: np.ndarray, Y: np.ndarray, test_array: np.ndarray, device: torch.device, surrogate: RTMSurrogate, models: dict) -> dict:
    torch.manual_seed(seed)
    n_bands, h, w = X.shape
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(test_array), min(N_SAMPLES, len(test_array)), replace=False)
    eval_labels_np = build_eval_labels(Y, test_array, h, w, sel)

    clean_scene = torch.from_numpy(X).to(device)
    eval_labels = torch.from_numpy(eval_labels_np).to(device)
    solar = (torch.rand(n_bands, generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.75).to(device)
    gen_atm_state = torch.tensor([GENERATION_ATM], device=device)
    eval_atm_state = torch.tensor([GENERATION_ATM], device=device)  # "clear" == generation condition

    adv_scenes = {}
    for source_name, source_model in models.items():
        attack = SACNetRTAAAttack(
            surrogate=surrogate, solar_irradiance=solar, epsilon=EPSILON, step_size=EPSILON / 5,
            n_steps=N_STEPS, phys_weights=PhysicalViabilityWeights(), mismatch_config=AtmosphericMismatchConfig(),
        )
        adv_scene, _ = attack.generate(classifier=source_model, clean_scene=clean_scene, eval_labels=eval_labels, atm_state=gen_atm_state)
        adv_scenes[source_name] = adv_scene

    seed_result = {"seed": seed}
    for source_name, adv_scene in adv_scenes.items():
        for target_name, target_model in models.items():
            acc = simulate_and_evaluate_scene(adv_scene, eval_labels, eval_atm_state, surrogate, solar, target_model)
            seed_result[f"src_{source_name}__tgt_{target_name}"] = 1.0 - acc
    return seed_result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X = np.load(DATA_DIR / "X.npy").astype(np.float32)
    Y = np.load(DATA_DIR / "Y.npy")
    test_array = np.load(DATA_DIR / "test_array.npy")
    n_bands = X.shape[0]

    sacnet = load_sacnet(n_bands=n_bands, n_classes=N_CLASSES, device=str(device))
    sacnet.load_state_dict(torch.load(CHECKPOINTS["sacnet"], map_location=device))
    sacnet.eval()

    s3anet = load_s3anet(n_bands=n_bands, n_classes=N_CLASSES, device=str(device))
    s3anet.load_state_dict(torch.load(CHECKPOINTS["s3anet"], map_location=device))
    s3anet.eval()

    models = {"sacnet": sacnet, "s3anet": s3anet}

    surrogate = RTMSurrogate.from_pretrained(RTM_CHECKPOINT, n_bands=n_bands).to(device)
    surrogate.eval()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"=== seed {seed}/{N_SEEDS-1} ===")
        result = run_one_seed(seed, X, Y, test_array, device, surrogate, models)
        all_results.append(result)
        for source_name in models:
            for target_name in models:
                v = result[f"src_{source_name}__tgt_{target_name}"]
                tag = "(white-box)" if source_name == target_name else "(BLACK-BOX TRANSFER)"
                print(f"  src={source_name:8s} tgt={target_name:8s} ASR={v:.4f} {tag}")

    with open("transferability_matrix_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n=== Transferability matrix, ASR mean +- s.d. across {N_SEEDS} seeds ===")
    header_label = "src / tgt"
    print(f"{header_label:12s}" + "".join(f"{t:>14s}" for t in models))
    for source_name in models:
        row = f"{source_name:12s}"
        for target_name in models:
            vals = np.array([r[f"src_{source_name}__tgt_{target_name}"] for r in all_results])
            row += f"{vals.mean():>8.4f}+-{vals.std(ddof=1):.3f}"
        print(row)

    print("\n=== White-box vs. black-box transfer, paired t-test ===")
    for source_name in models:
        for target_name in models:
            if source_name == target_name:
                continue
            white_vals = np.array([r[f"src_{target_name}__tgt_{target_name}"] for r in all_results])
            black_vals = np.array([r[f"src_{source_name}__tgt_{target_name}"] for r in all_results])
            diff = black_vals - white_vals
            _t, p = stats.ttest_rel(black_vals, white_vals)
            print(f"target={target_name}: white-box(src={target_name})={white_vals.mean():.4f}  "
                  f"black-box(src={source_name})={black_vals.mean():.4f}  diff={diff.mean():+.4f}  p={p:.5f}")

    print("\nSaved per-seed results to transferability_matrix_results.json")


if __name__ == "__main__":
    main()
