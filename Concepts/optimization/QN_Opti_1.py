import numpy as np
import pandas as pd
import os
from pydasa import Queue


# Set pandas display options to show 4 decimal places
pd.set_option('display.float_format', '{:.4f}'.format)


# Simple formatter for console output
def fmt(value):
    """Format a number to 4 decimal places for console output"""
    if isinstance(value, (int, float, np.number)):
        if np.isnan(value) or np.isinf(value):
            return str(value)
        return f"{value:.4f}"
    elif isinstance(value, np.ndarray):
        return np.array([fmt(x) for x in value])
    return value


# -----------------------------
# Load Data
# -----------------------------
def load(fname: str) -> pd.DataFrame:
    """Load configuration from a CSV file.
    CSV format:
        - node: <node_id>
        - miu: <mean_service_time>
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

def solve_jackson_network(miu, lambda0, n_servers, kap, P):
    """Solve open Jackson network using traffic equations"""
    Is = np.eye(len(miu))
    # print(I)
    lambdas = np.linalg.solve(Is - P.T, lambda0)
    # Format lambdas for console output
    print([fmt(x) for x in lambdas])

    # AKI voy!!! hacerlo generico para los modelos MM1, MMC, MM1K y MMCK
    # rho = lambdas / miu  # utilization
    rho = []
    # miu = []
    L = []
    Lq = []
    W = []
    Wq = []

    for la, m, n, k in zip(lambdas, miu, n_servers, kap):
        q = Queue(la, m, n, k)
        q.calculate_metrics()
        rho.append(q.rho)
        # miu.append(q.miu)
        L.append(q.avg_len)
        Lq.append(q.avg_len_q)
        W.append(q.avg_wait)
        Wq.append(q.avg_wait_q)
        # print(q)

    # Format rho values for console output
    print([fmt(r) for r in rho])
    if any(r >= 1.0 for r in rho):
        print("⚠️ Warning: System unstable (ρ >= 1 at some node).")

    # Calculate network-wide metrics
    network_metrics = calculate_network_metrics(lambdas,
                                                L,
                                                Lq,
                                                W,
                                                Wq,
                                                rho,
                                                miu)

    # Return individual node metrics
    return pd.DataFrame({
        "lambda": lambdas,
        "miu": miu,
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq
    }), network_metrics


def calculate_network_metrics(lambdas, L, Lq, W, Wq, rho=None, miu=None):
    """Calculate network-wide performance metrics.

    For a Jackson network, we calculate:
    1. L_network: Total average number of jobs in the system
    2. Lq_network: Total average number of jobs waiting in queues
    3. W_network: Average time a job spends in the system (weighted by arrival rates)
    4. Wq_network: Average time a job spends waiting in queues (weighted by arrival rates)

    Parameters:
    -----------
    lambdas : array-like
        Arrival rates at each node
    L, Lq, W, Wq : array-like
        Node-specific performance metrics
    rho : array-like, optional
        Utilization at each node
    miu : array-like, optional
        Service rates for fallback calculations

    Returns:
    --------
    dict
        Dictionary containing network-wide metrics
    """
    # Sum of all arrival rates (total throughput)
    total_lambda = np.sum(lambdas)

    # Total L and Lq are the sum of all node L and Lq values
    L_network = np.sum(L)
    Lq_network = np.sum(Lq)

    # For W and Wq, we need to consider the relative importance of each node
    # based on its arrival rate (throughput-weighted average)
    if total_lambda > 0:
        # Weighted averages based on relative throughput
        W_network = np.sum(np.multiply(W, lambdas)) / total_lambda
        Wq_network = np.sum(np.multiply(Wq, lambdas)) / total_lambda
    else:
        W_network = 0
        Wq_network = 0

    # Average utilization (arithmetic mean of node utilizations)
    avg_rho = np.mean(rho) if rho is not None and len(rho) > 0 else 0

    # Average service rate (arithmetic mean of node service rates)
    avg_miu = np.mean(miu) if miu is not None and len(miu) > 0 else 0

    return {
        "avg_miu": avg_miu,
        "avg_rho": avg_rho,
        "L_net": L_network,
        "Lq_net": Lq_network,
        "W_net": W_network,
        "Wq_net": Wq_network,
        "total_throughput": total_lambda
    }

# -----------------------------
# Optimization
# -----------------------------


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load configuration with mixed queue models
    cfg = load("qn_config_mixed.csv")
    print(cfg)
    miu = cfg["miu"].values
    lambda0 = cfg["lambda0"].values
    n_servers = cfg["s"].values
    kap = cfg["K"].values.astype(float)  # Convert to float array

    # Convert K=0 to infinite capacity
    for i in range(len(kap)):
        if kap[i] == 0:
            kap[i] = float('inf')

    # Convert string representations of arrays to actual numpy arrays
    # and create routing matrix P
    prob = []
    for pm_str in cfg["pm"].values:
        pm_values = pm_str.strip("[]").split(",")
        pm_values = [float(val) for val in pm_values]
        prob.append(pm_values)
    P = np.array((prob))

    # Solve the network analytically
    analyt_nd_metrics, analyt_net_metrics = solve_jackson_network(miu,
                                                                  lambda0,
                                                                  n_servers,
                                                                  kap,
                                                                  P)

    print("\n--- Analytical Jackson Network (Node Metrics) ---")
    print(analyt_nd_metrics)
    print("\n--- Analytical Jackson Network (Network-wide Metrics) ---")
    for key, value in analyt_net_metrics.items():
        print(f"{key}: {fmt(value)}")

    # Run optimization