"""Master script to run the full adversarial benchmark across all models, attacks, and datasets.
This script is designed to be run from the Colab notebook.
"""

import argparse
import json
import time
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path if needed
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from rtaa.data.hsi_dataset import load_hsi_cube, normalize_reflectance
from rtaa.eval.metrics import (
    overall_accuracy, average_accuracy, cohen_kappa, per_class_accuracy,
    spectral_angle_mapper, spectral_information_divergence, 
    physical_consistency_rate, attack_success_rate
)
from rtaa.eval.excel_writer import write_benchmark_results
from rtaa.attacks.baselines import fgsm_attack, pgd_attack, ifgsm_attack, ssfgsm_attack_full_scene
from rtaa.rtm.surrogate import RTMSurrogate
from rtaa.rtm.mismatch import AtmosphericMismatchConfig

DATASETS = {
    "PaviaU": {"bands": 103, "classes": 9},
    "IndianPines": {"bands": 200, "classes": 16},
    "Salinas": {"bands": 204, "classes": 16},
}

MODELS = ["HybridSN", "S3ANet", "SACNet", "SpectralFormer-pixelwise", "SpectralFormer-patchwise"]
ATTACKS = ["FGSM", "I-FGSM", "PGD", "SS-FGSM", "RTAA"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/content/drive/MyDrive/S3Anet_data")
    parser.add_argument("--out", type=str, default="benchmark_results.xlsx")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    results = []
    
    print("WARNING: This is a placeholder master script. In the complete version,")
    print("this script loads each model, loops over datasets, and runs each attack.")
    print("Since model dependencies (S3ANet, SACNet, SpectralFormer) require specific")
    print("repo setups, we output a structure that integrates with Colab.")
    
    # Mocking some results to verify Excel output works
    for ds_name, ds_info in DATASETS.items():
        n_classes = ds_info["classes"]
        for model_name in MODELS:
            for attack_name in ATTACKS:
                results.append({
                    "dataset": ds_name,
                    "model": model_name,
                    "attack": attack_name,
                    "OA": np.random.uniform(20, 90),
                    "Kappa": np.random.uniform(0.1, 0.8),
                    "AA": np.random.uniform(20, 90),
                    "class_accs": [np.random.uniform(0, 100) for _ in range(n_classes)],
                    "train_time": np.random.uniform(100, 2000),
                    "test_time": np.random.uniform(1, 10),
                    "total_time": np.random.uniform(101, 2010),
                    "SAM": np.random.uniform(1, 15),
                    "SID": np.random.uniform(0.001, 0.05),
                    "phys_consistency": np.random.uniform(10, 99),
                    "ASR": np.random.uniform(10, 95),
                })
                
    write_benchmark_results(results, args.out)
    print(f"Results written to {args.out}")

if __name__ == "__main__":
    main()
