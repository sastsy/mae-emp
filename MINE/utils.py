import os
import logging
from typing import Tuple, List, Dict, Any
from omegaconf import OmegaConf

import torch
import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_correlated_gaussians(
    num_samples: int,
    dim: int,
    rho: float,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    np.random.seed(seed)

    Z1 = np.random.normal(mean, std, (num_samples, dim))
    Z2 = np.random.normal(mean, std, (num_samples, dim))

    X = Z1
    Y = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32).to(device)

    return X_tensor, Y_tensor


def load_config(config_path: str = 'config.yaml') -> OmegaConf:
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        config = OmegaConf.load(config_path)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading or parsing YAML file {config_path} with OmegaConf: {e}")
        raise


def plot_mi_results(all_results: list, output_dir: str = 'plots'):
    logging.info("Generating plot of True vs Estimated Mutual Information...")
    plt.figure(figsize=(10, 6))

    plot_rhos = [r['rho'] for r in all_results]
    true_mis = [r['true_mi'] for r in all_results]
    estimated_mis = [r['estimated_mi'] for r in all_results]

    plt.plot(plot_rhos, true_mis, 'b-', label='True MI', linewidth=2)
    plt.plot(plot_rhos, estimated_mis, 'ro--', label='Estimated MI (unbiased)')

    plt.xlabel('Correlation coefficient (ρ)')
    plt.ylabel('Mutual Information')
    plt.title('True vs Estimated Mutual Information')
    plt.legend()
    plt.grid(True)
    plt.xticks(plot_rhos)

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'mi_estimation_results.png')
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")
    plt.show()


def plot_individual_rho_results(all_results: List[Dict[str, Any]], output_dir: str = 'plots'):
    logging.info("Generating individual plots for each rho...")
    os.makedirs(output_dir, exist_ok=True)

    for result in all_results:
        rho = result['rho']
        true_mi = result['true_mi']
        mi_history = result['mi_history']

        epochs = [item['epoch'] for item in mi_history]
        estimated_mis = [item['mi_estimate'] for item in mi_history]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, estimated_mis, 'b-o', label='Estimated MI')
        plt.axhline(y=true_mi, color='r', linestyle='--', label='True MI')

        plt.xlabel('Epoch')
        plt.ylabel('Mutual Information')
        plt.title(f'MI Estimation for Rho = {rho:.2f}')
        plt.legend()
        plt.grid(True)

        filename_rho = str(rho).replace('.', '_').replace('-', 'neg_')
        plot_path = os.path.join(output_dir, f'mi_estimation_rho_{filename_rho}.png')
        plt.savefig(plot_path)
        plt.close()

    logging.info("All individual plots generated.")