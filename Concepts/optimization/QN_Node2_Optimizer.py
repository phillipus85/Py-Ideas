import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pydasa import Queue
import networkx as nx

# Set pandas display options
pd.set_option('display.float_format', '{:.4f}'.format)


# Simple formatter for console output
def fmt(value):
    """Format a number to 4 decimal places for console output"""
    if isinstance(value, (int, float, np.number)):
        if np.isnan(value) or np.isinf(value):
            return str(value)
        return f"{value:.4f}"
    return value


# Load configuration from CSV
def load(fname):
    path = os.path.dirname(__file__)
    file_path = os.path.join(path, fname)
    df = pd.read_csv(file_path)
    return df


# Solve Jackson network
def solve_jackson_network(miu, lambda0, n_servers, kap, P, verbose=True):
    # Identity matrix
    Id = np.eye(len(miu))

    # Solve traffic equations to get arrival rates at each node
    lambdas = np.linalg.solve(Id - P.T, lambda0)

    if verbose:
        print("Node arrival rates (λ):", [fmt(x) for x in lambdas])

    # Calculate metrics for each node
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

    if verbose and any(r >= 1.0 for r in rho):
        print("⚠️ Warning: System unstable (ρ >= 1 at some node).")

    # Calculate network-wide metrics
    network_metrics = calculate_network_metrics(lambdas, L, Lq, W, Wq, rho, miu)

    # Return node metrics and network metrics
    return pd.DataFrame({
        "lambda": lambdas,
        "miu": miu,
        "rho": rho,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq
    }), network_metrics


# Calculate network-wide metrics
def calculate_network_metrics(lambdas, L, Lq, W, Wq, rho=None, miu=None):
    # Total throughput
    total_lambda = np.sum(lambdas)

    # Network-wide metrics
    L_network = np.sum(L)
    Lq_network = np.sum(Lq)

    # Weighted averages based on throughput
    if total_lambda > 0:
        W_network = np.sum(np.multiply(W, lambdas)) / total_lambda
        Wq_network = np.sum(np.multiply(Wq, lambdas)) / total_lambda
    else:
        W_network = 0
        Wq_network = 0

    # Average utilization and service rate
    avg_rho = np.mean(rho) if rho is not None else 0
    avg_miu = np.mean(miu) if miu is not None else 0

    return {
        "avg_rho": avg_rho,
        "avg_miu": avg_miu,
        "L_net": L_network,
        "Lq_net": Lq_network,
        "W_net": W_network,
        "Wq_net": Wq_network,
        "total_throughput": total_lambda,
    }


# Optimize queue network
def optimize_node_2(lambda0,
                    P_initial,
                    miu_initial,
                    n_servers,
                    kap,
                    miu_bounds=(6, 10),
                    prob_bounds=(0.1, 0.5)):
    """
    Optimize node 2's service rate and inbound routing probabilities
    """
    # Save initial configuration
    P = P_initial.copy()
    miu = miu_initial.copy()

    # Calculate initial metrics
    node_metrics_initial, net_metrics_initial = solve_jackson_network(
        miu, lambda0, n_servers, kap, P, verbose=False)

    print("\nInitial configuration:")
    print(f"  Node 2 service rate: {fmt(miu[2])}")
    print(f"  Node 0→2 probability: {fmt(P[0, 2])}")
    print(f"  Node 1→2 probability: {fmt(P[1, 2])}")
    print(f"  Overall waiting time: {fmt(net_metrics_initial['W_net'])}")

    # Parameters to optimize: [miu_node2, P_0_to_2, P_1_to_2]
    x0 = [miu[2], P[0, 2], P[1, 2]]

    # Set bounds for optimization variables
    bounds = [
        miu_bounds,      # Service rate bounds for node 2
        prob_bounds,     # Probability bounds for P[0,2]
        prob_bounds      # Probability bounds for P[1,2]
    ]

    # Define objective function to minimize overall waiting time
    def objective_function(params):
        # Create copies to modify
        current_miu = miu.copy()
        current_P = P.copy()

        # Update service rate for node 2
        current_miu[2] = params[0]

        # Update routing probabilities to node 2
        current_P[0, 2] = params[1]
        current_P[1, 2] = params[2]

        # Ensure row sums are <= 1 (exit probability = 1 - row_sum)
        for i in range(len(current_P)):
            row_sum = np.sum(current_P[i])
            if row_sum > 0.99:
                current_P[i] = current_P[i] / row_sum * 0.99

        try:
            # Calculate network metrics
            _, net_metrics = solve_jackson_network(
                current_miu, lambda0, n_servers, kap, current_P, verbose=False)

            # Return overall waiting time as the objective
            return net_metrics['W_net']
        except Exception:
            # Return high value if calculation fails
            return 1000.0

    # Run optimization
    result = minimize(
        objective_function,
        x0,
        method='L-BFGS-B',
        bounds=bounds
    )

    # Extract optimized parameters
    opt_params = result.x
    opt_miu = miu.copy()
    opt_P = P.copy()

    # Update with optimal values
    opt_miu[2] = opt_params[0]
    opt_P[0, 2] = opt_params[1]
    opt_P[1, 2] = opt_params[2]

    # Calculate metrics with optimized parameters
    node_metrics_opt, net_metrics_opt = solve_jackson_network(
        opt_miu, lambda0, n_servers, kap, opt_P, verbose=False)

    # Calculate improvement
    W_net_initial = net_metrics_initial['W_net']
    W_net_opt = net_metrics_opt['W_net']
    improvement = (W_net_initial - W_net_opt) / W_net_initial * 100 if W_net_initial > 0 else 0

    # Return results
    return {
        'initial': {
            'miu': miu,
            'P': P,
            'metrics': net_metrics_initial,
            'node_metrics': node_metrics_initial
        },
        'optimized': {
            'miu': opt_miu,
            'P': opt_P,
            'metrics': net_metrics_opt,
            'node_metrics': node_metrics_opt
        },
        'improvement': improvement,
        'success': result.success
    }


# Print optimization results
def print_optimization_results(result):
    """Print optimization results in a clear format"""
    print("\n" + "=" * 50)
    print("QUEUE NETWORK OPTIMIZATION RESULTS")
    print("=" * 50)
    
    initial = result['initial']
    optimized = result['optimized']
    
    # Print service rates
    print("\nService Rates (μ):")
    for i, (before, after) in enumerate(zip(initial['miu'], optimized['miu'])):
        marker = " ⭐" if i == 2 else ""
        print(f"  Node {i}: {fmt(before)} → {fmt(after)}{marker}")

    # Print routing probabilities
    print("\nRouting Probabilities:")
    print(f"  P[0,2]: {fmt(initial['P'][0, 2])} → {fmt(optimized['P'][0, 2])} ⭐")
    print(f"  P[1,2]: {fmt(initial['P'][1, 2])} → {fmt(optimized['P'][1, 2])} ⭐")

    # Print network metrics
    print("\nNetwork Metrics:")
    metrics = [
        ("W_net", "Overall Waiting Time"),
        ("L_net", "Overall Queue Length"),
        ("avg_rho", "Average Utilization"),
        ("total_throughput", "Total Throughput")
    ]

    for key, name in metrics:
        before = initial['metrics'][key]
        after = optimized['metrics'][key]
        change = (after - before) / before * 100 if before > 0 else 0
        direction = "▼" if change < 0 else "▲"
        print(f"  {name}: {fmt(before)} → {fmt(after)} ({fmt(change)}% {direction})")
    
    print(f"\nOptimization success: {result['success']}")
    print(f"Overall waiting time reduced by {fmt(result['improvement'])}%")
    print("=" * 50)

# Visualize the network
def visualize_network_comparison(before, after, lambda0):
    """Create a side-by-side visualization of network before and after optimization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create networks
    for ax, data, title in [
        (ax1, before, "Network Before Optimization"),
        (ax2, after, "Network After Optimization")
    ]:
        G = nx.DiGraph()
        P = data['P']
        miu = data['miu']
        n = len(miu)
        
        # Add nodes
        for i in range(n):
            G.add_node(i, external=lambda0[i], service=miu[i])
        
        # Add edges
        for i in range(n):
            for j in range(n):
                if P[i, j] > 0:
                    G.add_edge(i, j, weight=P[i, j])
        
        # Create layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        node_colors = ['lightblue' if i != 2 else 'lightgreen' for i in range(n)]
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, ax=ax)
        
        # Draw edges
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_weights, arrowsize=15, alpha=0.7, ax=ax)
        
        # Node labels
        labels = {i: f"Node {i}\nλ₀={lambda0[i]:.2f}\nμ={miu[i]:.2f}" for i in range(n)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)
        
        # Edge labels
        edge_labels = {(i, j): f"{P[i][j]:.2f}" for i, j in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
        
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("network_optimization_comparison.png")
    plt.close()
    print("Network visualization saved as: network_optimization_comparison.png")

# Compare metrics before and after optimization
def plot_metrics_comparison(result):
    """Plot comparison of key metrics before and after optimization"""
    # Extract metrics
    before = result['initial']['node_metrics']
    after = result['optimized']['node_metrics']
    
    # Create figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    
    # Metrics to plot
    metrics = ['rho', 'L', 'W', 'Wq']
    titles = ['Utilization (ρ)', 'Queue Length (L)', 'Waiting Time (W)', 'Queue Waiting Time (Wq)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        nodes = before.index
        axs[i].bar(nodes - 0.2, before[metric], width=0.4, label='Before')
        axs[i].bar(nodes + 0.2, after[metric], width=0.4, label='After')
        
        axs[i].set_title(title)
        axs[i].set_xlabel('Node')
        axs[i].set_xticks(nodes)
        axs[i].set_xticklabels([f'Node {i}' for i in nodes])
        axs[i].legend()
        
        # Highlight node 2
        axs[i].axvspan(1.5, 2.5, alpha=0.2, color='green')
    
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.close()
    print("Metrics comparison saved as: metrics_comparison.png")

# Main function
if __name__ == "__main__":
    # Load configuration
    cfg = load("qn_config_mixed.csv")
    print(cfg)
    
    # Extract parameters
    miu = cfg["miu"].values
    lambda0 = cfg["lambda0"].values
    n_servers = cfg["s"].values
    kap = cfg["K"].values.astype(float)
    
    # Convert K=0 to infinite capacity
    for i in range(len(kap)):
        if kap[i] == 0:
            kap[i] = float('inf')
    
    # Parse routing probabilities
    prob = []
    for pm_str in cfg["pm"].values:
        pm_values = pm_str.strip("[]").split(",")
        pm_values = [float(val) for val in pm_values]
        prob.append(pm_values)
    P = np.array(prob)
    
    # Solve the network analytically
    analyt_nd_metrics, analyt_net_metrics = solve_jackson_network(
        miu, lambda0, n_servers, kap, P)
    
    print("\n--- Analytical Jackson Network (Node Metrics) ---")
    print(analyt_nd_metrics)
    print("\n--- Analytical Jackson Network (Network-wide Metrics) ---")
    for key, value in analyt_net_metrics.items():
        print(f"{key}: {fmt(value)}")
    
    # Run optimization for node 2
    print("\n--- Optimizing Node 2 ---")
    opt_result = optimize_node_2(
        lambda0=lambda0,
        P_initial=P,
        miu_initial=miu,
        n_servers=n_servers,
        kap=kap,
        miu_bounds=(6, 10),      # Service rate between 6 and 10
        prob_bounds=(0.1, 0.5)   # Probabilities between 0.1 and 0.5
    )
    
    # Print optimization results
    print_optimization_results(opt_result)
    
    # Visualize the network before and after optimization
    visualize_network_comparison(
        opt_result['initial'],
        opt_result['optimized'],
        lambda0
    )
    
    # Plot metrics comparison
    plot_metrics_comparison(opt_result)
