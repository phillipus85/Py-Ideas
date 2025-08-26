import numpy as np
import pandas as pd
import os
from pydasa import Queue
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
    print(lambdas)

    # AKI voy!!! hacerlo generico para los modelos MM1, MMC, MM1K y MMCK
    # rho = lambdas / mu  # utilization
    rho = []
    L = []
    Lq = []
    W = []
    Wq = []

    for la, m in zip(lambdas, mu):
        q = Queue(la, m)
        q.calculate_metrics()
        rho.append(q.rho)
        L.append(q.avg_len)
        Lq.append(q.avg_len_q)
        W.append(q.avg_wait)
        Wq.append(q.avg_wait_q)
        # print(q)

    print(rho)
    if any(r >= 1.0 for r in rho):
        print("⚠️ Warning: System unstable (ρ >= 1 at some node).")

    # # M/M/1 metrics
    # L = rho / (1 - rho)
    # Lq = L - rho
    # W = L / lambdas
    # Wq = Lq / lambdas

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
# Simulation with SimPy
# -----------------------------


class QueueNode:
    def __init__(self, env, node_id, mu, P, results, sim_time):
        self.env = env
        self.node_id = node_id
        self.mu = mu
        self.P = P
        self.results = results
        self.sim_time = sim_time
        self.server = simpy.Resource(env, capacity=1)
        self.wait_times = []

    def service(self, job):
        service_time = random.expovariate(self.mu)
        yield self.env.timeout(service_time)
        self.wait_times.append(service_time)


def job_generator(env, node_id, rate, nodes, P, results, sim_time):
    """Generate external arrivals"""
    while True:
        interarrival = random.expovariate(rate)
        yield env.timeout(interarrival)
        env.process(job(env, node_id, nodes, P, results, sim_time))


def job(env, node_id, nodes, P, results, sim_time):
    current = node_id
    while True:
        node = nodes[current]
        with node.server.request() as req:
            start = env.now
            yield req
            yield env.process(node.service("job"))
            wait = env.now - start
            results[current].append(wait)

        # routing decision
        exit_prob = 1 - np.sum(P[current])
        if random.random() < exit_prob:
            break  # leave system
        else:
            # Normalize the probabilities to ensure they sum to 1
            probs = P[current] / np.sum(P[current])
            next_node = np.random.choice(range(len(P)), p=probs)
            current = next_node


def simulate_network(mu, lambda0, P, sim_time=5000):
    env = simpy.Environment()
    results = [[] for _ in range(len(mu))]
    nodes = [QueueNode(env, i, mu[i], P, results, sim_time)
             for i in range(len(mu))]

    # start external arrivals
    for i, rate in enumerate(lambda0):
        if rate > 0:
            env.process(job_generator(
                env, i, rate, nodes, P, results, sim_time))

    env.run(until=sim_time)

    # summarize
    avg_wait = [np.mean(r) if len(r) > 0 else 0 for r in results]
    return pd.DataFrame({
        "Node": range(len(mu)),
        "Avg_Wait_Sim": avg_wait,
        "Jobs_Served": [len(r) for r in results]
    })


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cfg = load("qn_config.csv")
    print(cfg)
    mu = cfg["mu"].values
    lambda0 = cfg["lambda0"].values

    # Convert string representations of arrays to actual numpy arrays

    # and create routing matrix P
    prob = []
    for pm_str in cfg["pm"].values:
        pm_values = pm_str.strip("[]").split(",")
        pm_values = [float(val) for val in pm_values]
        prob.append(pm_values)
    P = np.array((prob))

    # Solve the network
    analytical = solve_jackson_network(mu, lambda0, P)
    print(analytical)

    print("\n--- SimPy Simulation ---")
    simulation = simulate_network(mu, lambda0, P, sim_time=100000)
    print(simulation)

    # Compare both
    merged = analytical.copy()
    merged["Avg_Wait_Sim"] = simulation["Avg_Wait_Sim"]
    merged["Jobs_Served"] = simulation["Jobs_Served"]
    print("\n--- Comparison ---")
    print(merged)
