import numpy as np
import pandas as pd
import os
import random
import simpy


# -----------------------------
# Load Data
# -----------------------------
def load(fname: str) -> pd.DataFrame:
    """Load configuration from a CSV file.
    CSV format:
        - node: <node_id>
        - mu: <mean_service_time>
        - c: <service_channels>
        - K: <buffer_capacity | max_queue_length>
        - lambda0: <initial_arrival_rate>
        - L0: <initial_queue_length>
        - pm: <matrix_routing_probabilities>
    """

    path = os.path.dirname(__file__)
    file_path = os.path.join(path, fname)
    df = pd.read_csv(file_path)
    return df


# -----------------------------
# Analytical Solver (Jackson)
# -----------------------------
def solve_jackson_network(mu, lambda0, P):
    """Solve open Jackson network using traffic equations"""
    I = np.eye(len(mu))
    # print(I)
    lambdas = np.linalg.solve(I - P.T, lambda0)

    # AKI voy!!! hacerlo generico para los modelos MM1, MMC, MM1K y MMCK
    rho = lambdas / mu  # utilization
    print(rho)
    if np.any(rho >= 1):
        print("⚠️ Warning: System unstable (ρ >= 1 at some node).")

    # M/M/1 metrics
    L = rho / (1 - rho)
    Lq = L - rho
    W = L / lambdas
    Wq = Lq / lambdas

    return pd.DataFrame({
        "lambda": lambdas,
        "mu": mu,
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq
    })


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cfg = load("qn_config.csv")
    mu = cfg["mu"].prob
    lambda0 = cfg["lambda0"].prob

    # Convert string representations of arrays to actual numpy arrays

    # and create routing matrix P
    prob = []
    for pm_str in cfg["pm"].values:
        pm_values = pm_str.strip("[]").split(",")
        pm_values = [float(val) for val in pm_values]
        prob.append(pm_values)
    P = np.array((prob))

    # Solve the network
    results = solve_jackson_network(mu, lambda0, P)
    print(results)
