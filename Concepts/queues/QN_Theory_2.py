import numpy as np
import pandas as pd
import os
from pydasa import Queue
import random
import simpy


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
    L = []
    Lq = []
    W = []
    Wq = []

    for la, m, n, k in zip(lambdas, miu, n_servers, kap):
        q = Queue(la, m, n, k)
        q.calculate_metrics()
        rho.append(q.rho)
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
    network_metrics = calculate_network_metrics(lambdas, L, Lq, W, Wq, rho)

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


def calculate_network_metrics(lambdas, L, Lq, W, Wq, rho=None):
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

    return {
        "L_net": L_network,
        "Lq_net": Lq_network,
        "W_net": W_network,
        "Wq_net": Wq_network,
        "avg_rho": avg_rho,
        "total_throughput": total_lambda
    }

# -----------------------------
# Simulation with SimPy
# -----------------------------


class QueueNode:
    def __init__(self, env, node_id, miu, s, K, P, results, sim_time):
        self.env = env
        self.node_id = node_id
        self.miu = miu  # Service rate per server
        self.s = s      # Number of servers
        self.K = K      # System capacity (queue + service)
        self.P = P
        self.results = results
        self.sim_time = sim_time

        # Use Resource with capacity equal to number of servers
        self.server = simpy.Resource(env, capacity=s)

        # Track when jobs are blocked due to capacity
        self.blocked_jobs = 0

        # For tracking queue and system metrics
        self.service_times = []  # Track service times
        self.queue_times = []    # Time spent only in queue
        self.system_times = []   # Total time in system (queue + service)

        # For time-weighted L and Lq calculation (event-driven)
        self.queue_len_data = []     # (length, time_delta) pairs
        self.system_len_data = []    # (length, time_delta) pairs
        self.last_event_time = 0
        self.in_queue = 0
        self.in_service = 0
        self.current_queue_length = 0
        self.current_system_length = 0

    def is_full(self):
        """Check if the system is at capacity"""
        return (self.in_queue + self.in_service) >= self.K

    def record_state_change(self, env):
        """Record a state change in the system"""
        current_time = env.now
        time_delta = current_time - self.last_event_time

        if time_delta > 0:
            # Record the previous state duration
            self.queue_len_data.append((self.current_queue_length, time_delta))
            self.system_len_data.append((self.current_system_length, time_delta))

        # Update current state
        self.current_queue_length = self.in_queue
        self.current_system_length = self.in_queue + self.in_service
        self.last_event_time = current_time

    def service(self, job):
        service_time = random.expovariate(self.miu)
        self.service_times.append(service_time)
        yield self.env.timeout(service_time)


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

        # Check if the system is at capacity (M/M/*/K model)
        if node.is_full():
            # Job is blocked/lost due to capacity constraint
            node.blocked_jobs += 1
            break  # Job leaves the network

        # Update queue length when job arrives
        node.in_queue += 1
        node.record_state_change(env)  # Record state change at arrival

        with node.server.request() as req:
            # Time when request is made
            queue_start = env.now

            # Wait until server is available
            yield req

            # Update queue length and track queue waiting time
            queue_time = env.now - queue_start
            node.queue_times.append(queue_time)
            node.in_queue -= 1
            node.in_service += 1  # Track actual number of jobs in service
            node.record_state_change(env)  # Record state change at service start

            # Service process
            yield env.process(node.service("job"))

            # Calculate total time in system
            total_time = env.now - arrival_time
            node.system_times.append(total_time)
            results[current].append(total_time)

            # Update system metrics
            node.in_service -= 1  # One job less in service
            node.record_state_change(env)  # Record state change at departure

        # routing decision
        exit_prob = 1 - np.sum(P[current])
        if random.random() < exit_prob:
            break  # leave system
        else:
            # Normalize the probabilities to ensure they sum to 1
            probs = P[current] / np.sum(P[current])
            next_node = np.random.choice(range(len(P)), p=probs)
            current = next_node


def simulate_network(miu, lambda0, P, s=None, K=None, sim_time=5000):
    """
    Simulate open queueing network using SimPy

    Parameters:
    -----------
    miu : array-like
        Service rates for each node
    lambda0 : array-like
        External arrival rates for each node
    P : array-like
        Routing probability matrix
    s : array-like, optional
        Number of servers for each node (default=1 for all nodes)
    K : array-like, optional
        System capacity for each node (default=inf for all nodes)
    sim_time : float
        Simulation duration
    """
    # Set default values if not provided
    if s is None:
        s = [1] * len(miu)
    if K is None:
        K = [float('inf')] * len(miu)

    env = simpy.Environment()
    results = [[] for _ in range(len(miu))]

    # Create nodes with multiple servers and capacity limits
    nodes = []
    for i in range(len(miu)):
        node = QueueNode(env, i, miu[i], s[i], K[i], P, results, sim_time)
        nodes.append(node)

    # Initialize state recording for each node
    for node in nodes:
        node.record_state_change(env)

    # start external arrivals
    for i, rate in enumerate(lambda0):
        if rate > 0:
            env.process(job_generator(
                env, i, rate, nodes, P, results, sim_time))

    env.run(until=sim_time)

    # Final state recording for accurate time-weighted statistics
    for node in nodes:
        # Record the final state duration
        time_delta = sim_time - node.last_event_time
        if time_delta > 0:
            node.queue_len_data.append((node.current_queue_length, time_delta))
            node.system_len_data.append((node.current_system_length, time_delta))

    # Calculate metrics for each node
    sim_metrics = []
    for i, node in enumerate(nodes):
        # Number of jobs served and blocked
        jobs_served = len(node.system_times)
        total_arrivals = jobs_served + node.blocked_jobs

        # Determine queue model type
        if node.s == 1:
            if node.K == float('inf'):
                model_type = "M/M/1"
            else:
                model_type = f"M/M/1/{node.K}"
        else:
            if node.K == float('inf'):
                model_type = f"M/M/{node.s}"
            else:
                model_type = f"M/M/{node.s}/{node.K}"

        # Print raw data for debugging
        print(f"Node {i} queue length data points: {len(node.queue_len_data)}")
        # Format sample data to 4 decimal places
        formatted_data = []
        for length, time_val in node.queue_len_data[:5]:
            if isinstance(time_val, (float, np.number)):
                formatted_data.append((length, float(fmt(time_val))))
            else:
                formatted_data.append((length, time_val))
        print(f"Sample queue length data: {formatted_data}")

        # Calculate time-average L (system length)
        total_time = sum(time for _, time in node.system_len_data) or sim_time
        L_sim = sum(length * time for length, time in node.system_len_data) / total_time if node.system_len_data else 0

        # Calculate time-average Lq (queue length)
        total_q_time = sum(time for _, time in node.queue_len_data) or sim_time
        Lq_sim = sum(length * time for length, time in node.queue_len_data) / total_q_time if node.queue_len_data else 0

        # Calculate W (time in system) and Wq (time in queue)
        W_sim = np.mean(node.system_times) if node.system_times else 0
        Wq_sim = np.mean(node.queue_times) if node.queue_times else 0

        # Calculate lambda (arrival rate) and rho (utilization)
        lambda_sim = jobs_served / sim_time

        # For multi-server queues, utilization is per server
        rho_sim = min(1.0, lambda_sim / (node.s * miu[i]))

        # Validate with Little's Law
        L_from_littles = lambda_sim * W_sim
        Lq_from_littles = lambda_sim * Wq_sim

        # Calculate blocking probability for finite capacity queues
        blocking_prob = node.blocked_jobs / total_arrivals if total_arrivals > 0 else 0

        sim_metrics.append({
            "Node": i,
            "Model": model_type,
            "lambda_sim": lambda_sim,
            "rho_sim": rho_sim,
            "L_sim": L_sim,
            "Lq_sim": Lq_sim,
            "W_sim": W_sim,
            "Wq_sim": Wq_sim,
            "L_littles": L_from_littles,
            "Lq_littles": Lq_from_littles,
            "Jobs_Served": jobs_served,
            "Jobs_Blocked": node.blocked_jobs,
            "Blocking_Prob": blocking_prob
        })

    # Calculate network-wide metrics
    lambda_sims = [metrics["lambda_sim"] for metrics in sim_metrics]
    L_sims = [metrics["L_sim"] for metrics in sim_metrics]
    Lq_sims = [metrics["Lq_sim"] for metrics in sim_metrics]
    W_sims = [metrics["W_sim"] for metrics in sim_metrics]
    Wq_sims = [metrics["Wq_sim"] for metrics in sim_metrics]

    network_metrics = calculate_network_metrics(
        lambda_sims, L_sims, Lq_sims, W_sims, Wq_sims,
        [metrics["rho_sim"] for metrics in sim_metrics]
    )

    return pd.DataFrame(sim_metrics), network_metrics


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load configuration with mixed queue models
    cfg = load("qn_config_v2.csv")
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

    # Run simulation
    print("\n--- SimPy Simulation ---")
    sim_nd_metrics, simulation_network_metrics = simulate_network(
        miu, lambda0, P, s=n_servers, K=kap, sim_time=50000)
    print(sim_nd_metrics)

    print("\n--- Simulation Network-wide Metrics ---")
    for key, value in simulation_network_metrics.items():
        print(f"{key}: {fmt(value)}")

    # Compare both
    print("\n--- Comparison of Node Metrics (Analytical vs Simulation) ---")
    # Create a comparison dataframe
    comparison = pd.DataFrame()
    comparison["Node"] = sim_nd_metrics["Node"]
    comparison["Model"] = sim_nd_metrics["Model"]

    # Add analytical metrics
    comparison["lambda_theo"] = analyt_nd_metrics["lambda"].values
    comparison["rho_theo"] = analyt_nd_metrics["rho"].values
    comparison["L_theo"] = analyt_nd_metrics["L"].values
    comparison["Lq_theo"] = analyt_nd_metrics["Lq"].values
    comparison["W_theo"] = analyt_nd_metrics["W"].values
    comparison["Wq_theo"] = analyt_nd_metrics["Wq"].values

    # Add simulation metrics
    comparison["lambda_sim"] = sim_nd_metrics["lambda_sim"]
    comparison["rho_sim"] = sim_nd_metrics["rho_sim"]
    comparison["L_sim"] = sim_nd_metrics["L_sim"]
    comparison["Lq_sim"] = sim_nd_metrics["Lq_sim"]
    comparison["W_sim"] = sim_nd_metrics["W_sim"]
    comparison["Wq_sim"] = sim_nd_metrics["Wq_sim"]
    # comparison["L_littles"] = sim_nd_metrics["L_littles"]
    # comparison["Lq_littles"] = sim_nd_metrics["Lq_littles"]
    comparison["Jobs_Served"] = sim_nd_metrics["Jobs_Served"]
    comparison["Jobs_Blocked"] = sim_nd_metrics["Jobs_Blocked"]
    comparison["Blocking_Prob"] = sim_nd_metrics["Blocking_Prob"]

    # Set pandas display options to show all columns with proper formatting
    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.expand_frame_repr', True, 'display.precision', 4):
        print(comparison)

    # Calculate percentage errors
    print("\n--- Percentage Errors (|Analytical - Simulation|/Analytical * 100%) ---")
    error_df = pd.DataFrame()
    error_df["Node"] = comparison["Node"]

    for metric in ["lambda", "rho", "L", "Lq", "W", "Wq"]:
        error_df[f"{metric}_err"] = 100 * abs(comparison[f"{metric}_theo"] - comparison[f"{metric}_sim"]) / comparison[f"{metric}_theo"]

    # Add Little's Law validation errors
    # error_df["L_littles_err"] = 100 * abs(comparison["L_theo"] - comparison["L_littles"]) / comparison["L_theo"]
    # error_df["Lq_littles_err"] = 100 * abs(comparison["Lq_theo"] - comparison["Lq_littles"]) / comparison["Lq_theo"]

    print(error_df)

    # Compare network-wide metrics
    print("\n--- Network-wide Metrics Comparison ---")
    network_comparison = pd.DataFrame(columns=["Metric", "Theoretical", "Simulation", "Error %"])

    for key in ["L_net", "Lq_net", "W_net", "Wq_net", "avg_rho", "total_throughput"]:
        if key in analyt_net_metrics and key in simulation_network_metrics:
            analytical_value = analyt_net_metrics[key]
            simulation_value = simulation_network_metrics[key]
            error = 100 * abs(analytical_value - simulation_value) / analytical_value if analytical_value != 0 else 0

            network_comparison.loc[len(network_comparison)] = [
                key,
                fmt(analytical_value),
                fmt(simulation_value),
                fmt(error)
            ]

    print(network_comparison)
