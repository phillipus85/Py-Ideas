import numpy as np
import pandas as pd
import os
import random
import simpy


# -----------------------------
# Load Data
# -----------------------------
def load_queue_data(service_file, routing_file):
    """Load service rates and routing matrix from CSV files"""
    path = os.path.dirname(__file__)
    file_path = os.path.join(path, service_file)
    services = pd.read_csv(file_path)
    mu = services['mu'].to_numpy(dtype=float)
    lambda0 = services['lambda0'].to_numpy(dtype=float)

    file_path = os.path.join(path, routing_file)
    P = pd.read_csv(file_path, header=None).to_numpy(dtype=float)
    return mu, lambda0, P


# -----------------------------
# Analytical Solver (Jackson)
# -----------------------------
def solve_jackson_network(mu, lambda0, P):
    """Solve open Jackson network using traffic equations"""
    Is = np.eye(len(mu))
    # print(I)
    lambdas = np.linalg.solve(Is - P.T, lambda0)

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
        self.queue_times = []  # Track time spent waiting in queue only
        self.service_times = []  # Track service times

    def service(self, job):
        service_time = random.expovariate(self.mu)
        self.service_times.append(service_time)
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
        arrival_time = env.now

        with node.server.request() as req:
            # Start measuring queue time
            queue_start = env.now
            yield req
            # Calculate time spent in queue only
            queue_time = env.now - queue_start
            node.queue_times.append(queue_time)

            # Start measuring service time
            # service_start = env.now
            yield env.process(node.service("job"))
            # Total time in system
            wait = env.now - arrival_time
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
    metrics = []
    for i, node in enumerate(nodes):
        # Calculate metrics
        W_sim = np.mean(results[i]) if results[i] else 0
        Wq_sim = np.mean(node.queue_times) if node.queue_times else 0
        jobs_served = len(results[i])
        lambda_sim = jobs_served / sim_time

        # Calculate rho using average service time
        avg_service_time = np.mean(node.service_times) if node.service_times else 0
        rho_sim = lambda_sim * avg_service_time if lambda_sim > 0 else 0

        # Little's Law: L = λW, Lq = λWq
        L_sim = lambda_sim * W_sim
        Lq_sim = lambda_sim * Wq_sim

        metrics.append({
            "Node": i,
            "W_Sim": W_sim,
            "Wq_Sim": Wq_sim,
            "L_Sim": L_sim,
            "Lq_Sim": Lq_sim,
            "rho_Sim": rho_sim,
            "lambda_Sim": lambda_sim,
            "Jobs_Served": jobs_served
        })

    return pd.DataFrame(metrics)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    mu, lambda0, P = load_queue_data("service_rates.csv", "routing_matrix.csv")

    print("\n--- Analytical Jackson Network ---")
    analytical = solve_jackson_network(mu, lambda0, P)
    print(analytical)

    print("\n--- SimPy Simulation ---")
    simulation = simulate_network(mu, lambda0, P, sim_time=10000)
    print(simulation)

    # Compare both
    merged = analytical.copy()
    merged["W_Sim"] = simulation["W_Sim"]
    merged["Jobs_Served"] = simulation["Jobs_Served"]
    print("\n--- Comparison ---")
    print(merged)
