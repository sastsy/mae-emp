import os
import click
import random
import logging
from tqdm import tqdm
from typing import List, Dict, Any

import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt

from utils import (
    load_config,
    plot_mi_results,
    plot_individual_rho_results,
    generate_correlated_gaussians,
)
from mine import MINE, EMA, true_mutual_information


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    logging.info(f"Random seed set to {seed}")


def compute_unbiased_mine_loss(
    model: MINE,
    x: torch.Tensor,
    y: torch.Tensor,
    y_marginal: torch.Tensor,
    ema: EMA,
) -> torch.Tensor:
    
    joint_output = model(x, y)
    E_joint = torch.mean(joint_output)

    marginal_output = model(x, y_marginal)
    E_exp_marginal = torch.mean(torch.exp(marginal_output))
    ema_val = ema.update(E_exp_marginal).detach()

    loss = -(E_joint - E_exp_marginal / ema_val)

    return loss

def compute_biased_mine_loss(
    model: MINE,
    x: torch.Tensor,
    y: torch.Tensor,
    y_marginal: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    joint_outputs = model(x, y)
    marginal_outputs = model(x, y_marginal)

    log_denom_biased = torch.logsumexp(marginal_outputs, dim=0) - torch.log(
        torch.tensor(marginal_outputs.size(0), device=device, dtype=torch.float32)
    )
    mi_lower_bound = joint_outputs.mean() - log_denom_biased
    loss = -mi_lower_bound

    return loss

def estimate_mi_unbiased(model: MINE, x: torch.Tensor, y: torch.Tensor, ema: EMA) -> float:
    model.eval()
    with torch.no_grad():
        E_T = torch.mean(model(x, y))
        ema_value = torch.tensor(ema.value, device=x.device, dtype=torch.float32)
        log_Z_ema = torch.log(ema_value)
        mi_estimate = E_T - log_Z_ema
    model.train()

    return mi_estimate.item()

def train_MINE(
    model: MINE,
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    n_epochs: int,
    batch_size: int,
    device: torch.device,
    learning_rate: float = 1e-4,
    ema_decay: float = 0.01,
    biased_loss: bool = False,
    logging_steps: int = 20,
) -> EMA:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    n_samples = X_data.shape[0]

    ema = EMA(decay=ema_decay)
    mi_estimates_history = []

    model.train()

    for epoch in range(n_epochs):
        permutation = torch.randperm(n_samples, device=device)
        X_shuffled = X_data[permutation]
        Y_shuffled = Y_data[permutation]

        Y_marginal_shuffled = Y_shuffled[torch.randperm(n_samples, device=device)]

        for batch_idx in range(0, n_samples, batch_size):
            optimizer.zero_grad()

            X_batch = X_shuffled[batch_idx:min(batch_idx + batch_size, n_samples)]
            Y_batch = Y_shuffled[batch_idx:min(batch_idx + batch_size, n_samples)]
            Y_marginal_batch = Y_marginal_shuffled[batch_idx:min(batch_idx + batch_size, n_samples)]

            if not biased_loss:
                loss = compute_unbiased_mine_loss(model, X_batch, Y_batch, Y_marginal_batch, ema)
            else:
                loss = compute_biased_mine_loss(model, X_batch, Y_batch, Y_marginal_batch, device)

            loss.backward()
            optimizer.step()

        if epoch % logging_steps == 0 or epoch == n_epochs - 1:
            mi_estimate = estimate_mi_unbiased(model, X_data, Y_data, ema)
            logging.info(f"Epoch {epoch}/{n_epochs}, MI (unbiased estimate): {mi_estimate:.4f}")
            mi_estimates_history.append({'epoch': epoch, 'mi_estimate': mi_estimate})

    return ema, mi_estimates_history

def train_MINE_for_rhos(
    rhos: List[float],
    num_samples: int,
    dim: int,
    n_epochs: int,
    batch_size: int,
    device: torch.device,
    learning_rate: float,
    ema_decay: float,
    biased_loss: bool,
    model_hidden_dim: int,
    seed: int,
    logging_steps: int,
) -> List[Dict[str, Any]]:
    results = []
    for rho in tqdm(rhos, desc="Training MINE for different rhos..."):
        print(f"\nRho = {rho}")
        X, Y = generate_correlated_gaussians(
            num_samples=num_samples,
            dim=dim,
            rho=rho,
            device=device,
            seed=seed,
        )

        model = MINE(input_dim=dim, hidden_dim=model_hidden_dim).to(device)

        final_ema, mi_history_for_rho = train_MINE(
            model=model,
            X_data=X,
            Y_data=Y,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            learning_rate=learning_rate,
            ema_decay=ema_decay,
            biased_loss=biased_loss,
            logging_steps=logging_steps,
        )

        mi_estimate = estimate_mi_unbiased(model, X, Y, final_ema)
        true_mi = true_mutual_information(rho, dim)

        results.append({
            'rho': rho,
            'true_mi': true_mi,
            'estimated_mi': mi_estimate,
            'mi_history': mi_history_for_rho,
        })

    return results

@click.command()
@click.option("--config_path", default="MINE/train_config.yaml")
def main(config_path: str):
    config = load_config(config_path)

    # -- Parameters from config ---
    num_samples = config.data_generation.get('num_samples', 20000)
    dim = config.data_generation.get('dim', 20)
    rhos_start = config.data_generation.get('rhos_start', -0.9)
    rhos_end = config.data_generation.get('rhos_end', 0.9)
    rhos_num = config.data_generation.get('rhos_num', 5)

    model_hidden_dim = config.model.get('hidden_dim', 128)

    n_epochs = config.training.get('n_epochs', 50)
    batch_size = config.training.get('batch_size', 64)
    learning_rate = config.training.get('learning_rate', 1e-4)
    ema_decay = config.training.get('ema_decay', 0.01)
    biased_loss = config.training.get('biased_loss', False)
    logging_steps = config.training.get('logging_steps', 10)

    plot_results = config.evaluation.get('plot_results', True)

    seed = config.get('seed', 8888)
    set_seed(seed)

    # --- Setting device ---
    device_str = config.get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Training MINE for different rhos ---
    # rhos = np.linspace(rhos_start, rhos_end, rhos_num)
    rhos = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

    logging.info("Starting MINE training for various correlation coefficients...")
    all_results = train_MINE_for_rhos(
        rhos=rhos,
        num_samples=num_samples,
        dim=dim,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        learning_rate=learning_rate,
        ema_decay=ema_decay,
        biased_loss=biased_loss,
        model_hidden_dim=model_hidden_dim,
        seed=seed,
        logging_steps=logging_steps,
    )
    logging.info("MINE training completed.")

    if plot_results:
        plot_mi_results(all_results)
        plot_individual_rho_results(all_results)
    else:
        logging.info("Plotting results skipped.")

    logging.info("Training finished.")


if __name__ == "__main__":
    main()
