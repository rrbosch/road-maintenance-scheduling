"""Static traffic assignment (the inner loop of the whole optimizer).

`TrafficNetwork` solves the **user-equilibrium Traffic Assignment Problem** (TAP): given a road
network and an origin-destination demand matrix, find the link flows in which no traveller can
reduce their travel time by switching route (Wardrop's first principle). The solver is
**Conjugate Frank-Wolfe (CFW)** — a conjugate-gradient acceleration of Frank-Wolfe:

  1. with the current link costs, route every OD pair onto its shortest path
     (`loadAON_array`, "all-or-nothing") to get a target flow pattern ``x_bar``;
  2. take a conjugate combination of the AON descent direction and the previous direction;
  3. line-search the optimal step ``alpha`` along it and move the flows;
  4. recompute link costs via the BPR function and repeat until the relative gap is small.

Shortest paths use a Numba-jitted Dijkstra (the `@njit` functions below); link cost is the BPR
volume-delay function. This module is performance-critical: it is called once per distinct
"ongoing-projects" scenario (a crippled copy of the network) and is the dominant cost of the
whole multi-objective search, which is why the lower-bound regressor exists to avoid it.
"""
import time
import warnings
from collections import defaultdict

import numpy as np
from numba import njit
from scipy.optimize import fsolve

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- BPR (Bureau of Public Roads) volume-delay parameters ---
# travel_time = free_flow_time * (1 + ALPHA * (flow / capacity) ** BETA)
# ALPHA=0.15, BETA=4 are the standard BPR calibration used throughout the TAP literature.
BPR_ALPHA = 0.15
BPR_BETA = 4
# Links with (near-)zero capacity are treated as effectively impassable (huge cost).
MIN_CAPACITY = 1e-3

# --- "crippling" a link to model a road under construction ---
# A closed/works link keeps its endpoints in the graph but is made very unattractive: its
# free-flow time is inflated and its capacity slashed, so equilibrium routes mostly avoid it
# without disconnecting the network.
CRIPPLE_FFT_MULTIPLIER = 1000   # free-flow time x1000
CRIPPLE_CAPACITY_FACTOR = 0.1   # capacity x0.1


@njit
def dijkstra_numba_optimized(edge_from, edge_to, edge_costs, origin_idx, n_nodes):
    """Fast Numba-based Dijkstra implementation using edge arrays with adjacency list optimization."""
    distances = np.full(n_nodes, np.inf, dtype=np.float64)
    predecessors = np.full(n_nodes, -1, dtype=np.int32)
    visited = np.zeros(n_nodes, dtype=np.bool_)

    distances[origin_idx] = 0.0

    # Pre-build adjacency lists for faster neighbor lookup
    max_edges_per_node = 0
    edge_counts = np.zeros(n_nodes, dtype=np.int32)

    # Count edges per node
    for edge_idx in range(len(edge_from)):
        edge_counts[edge_from[edge_idx]] += 1
        if edge_counts[edge_from[edge_idx]] > max_edges_per_node:
            max_edges_per_node = edge_counts[edge_from[edge_idx]]

    # Build adjacency lists
    adj_neighbors = np.full((n_nodes, max_edges_per_node), -1, dtype=np.int32)
    adj_costs = np.full((n_nodes, max_edges_per_node), np.inf, dtype=np.float64)
    edge_counts.fill(0)  # Reset for building

    for edge_idx in range(len(edge_from)):
        from_node = edge_from[edge_idx]
        to_node = edge_to[edge_idx]
        cost = edge_costs[edge_idx]

        adj_neighbors[from_node, edge_counts[from_node]] = to_node
        adj_costs[from_node, edge_counts[from_node]] = cost
        edge_counts[from_node] += 1

    for _ in range(n_nodes):
        # Find unvisited node with minimum distance
        current = -1
        min_dist = np.inf

        for i in range(n_nodes):
            if not visited[i] and distances[i] < min_dist:
                min_dist = distances[i]
                current = i

        if current == -1 or min_dist == np.inf:
            break

        visited[current] = True

        # Update distances to neighbors using adjacency list
        for i in range(edge_counts[current]):
            neighbor = adj_neighbors[current, i]
            if neighbor != -1 and not visited[neighbor]:
                edge_cost = adj_costs[current, i]
                new_dist = distances[current] + edge_cost
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    predecessors[neighbor] = current

    return distances, predecessors


@njit
def reconstruct_path_numba_optimized(predecessors, origin_idx, dest_idx):
    """Reconstruct path using pure NumPy operations."""
    # Check if destination is reachable
    if dest_idx == origin_idx:
        path = np.empty(1, dtype=np.int32)
        path[0] = origin_idx
        return path

    if predecessors[dest_idx] == -1:
        # Return empty path with size 0 (Numba-compatible)
        path = np.empty(0, dtype=np.int32)
        return path

    # Count path length first
    path_length = 0
    current = dest_idx
    temp_current = current

    # Count nodes in path
    while temp_current != -1 and path_length < 1000:  # Safety limit
        path_length += 1
        if temp_current == origin_idx:
            break
        temp_current = predecessors[temp_current]

    if temp_current != origin_idx or path_length == 0:
        # Return empty path
        path = np.empty(0, dtype=np.int32)
        return path

    # Build path array
    path = np.empty(path_length, dtype=np.int32)
    current = dest_idx

    for i in range(path_length):
        path[path_length - 1 - i] = current
        if current == origin_idx:
            break
        current = predecessors[current]

    return path


@njit
def add_path_flows_numba(path, demand, edge_from, edge_to, x_bar):
    """Add demand flow to edges along path using Numba optimization."""
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]

        # Find edge - optimized linear search
        for edge_idx in range(len(edge_from)):
            if edge_from[edge_idx] == from_node and edge_to[edge_idx] == to_node:
                x_bar[edge_idx] += demand
                break


@njit
def BPR_numba(fft, flow, capacity):
    """BPR travel time for a single link: fft * (1 + ALPHA*(flow/capacity)**BETA)."""
    if capacity < MIN_CAPACITY:
        return np.finfo(np.float32).max
    return fft * (1 + BPR_ALPHA * ((flow / capacity) ** BPR_BETA))

@njit
def BPR_vectorized(fft_array, flow_array, capacity_array):
    """BPR travel time for every link (array form of `BPR_numba`)."""
    n = len(fft_array)
    costs = np.empty(n)
    for i in range(n):
        if capacity_array[i] < MIN_CAPACITY:
            costs[i] = np.finfo(np.float32).max
        else:
            costs[i] = fft_array[i] * (1 + BPR_ALPHA * ((flow_array[i] / capacity_array[i]) ** BPR_BETA))
    return costs


@njit
def BPR_derivative_vectorized(fft_array, flow_array, capacity_array):
    """d(travel time)/d(flow) per link = fft * ALPHA*BETA * flow**(BETA-1) / capacity**BETA.

    Used for gradient-based line search / conjugacy calculations.
    """
    n = len(fft_array)
    derivatives = np.empty(n)
    for i in range(n):
        if capacity_array[i] < MIN_CAPACITY:
            derivatives[i] = 0.0
        else:
            derivatives[i] = (fft_array[i] * BPR_ALPHA * BPR_BETA
                              * (flow_array[i] ** (BPR_BETA - 1)) / (capacity_array[i] ** BPR_BETA))
    return derivatives


class TrafficNetwork:
    """Conjugate Frank-Wolfe implementation for Traffic Assignment Problem."""

    def __init__(self, links, nodes, trips):
        self.trips = trips
        self._cost: float = np.nan      # last computed Total System Travel Time (set by assignment_loop)
        self.feasible: bool = True
        self.maxIter = 100              # hard cap on CFW iterations (give up if not converged)
        self.maxTime = 60              # hard cap in seconds on a single assignment
        self.iteration_number = 0
        self.ongoing_projects = []      # links crippled on this network copy (for bookkeeping/plots)
        self.verbose = False

        # CFW-specific parameters
        self.beta_method = 'polak-ribiere'  # conjugacy formula: 'polak-ribiere' or 'fletcher-reeves'
        self.restart_threshold = 0.1  # drop conjugacy and fall back to plain FW if the conjugate
                                      # direction is poorly aligned (cosine angle < threshold)

        # Convert to array representation
        self._initialize_arrays(links, nodes, trips)

    def _initialize_arrays(self, links, nodes, trips):
        """Convert NetworkX representation to pure arrays."""
        # Node mapping
        self.nodes = sorted(nodes.index.values)
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.node_pos = nodes[['X', 'Y']].to_dict()

        # Edge arrays
        self.n_edges = len(links)
        self.edge_from = np.zeros(self.n_edges, dtype=np.int32)
        self.edge_to = np.zeros(self.n_edges, dtype=np.int32)
        self.edge_fft = np.zeros(self.n_edges, dtype=np.float64)
        self.edge_capacity = np.zeros(self.n_edges, dtype=np.float64)
        self.edge_flow = np.zeros(self.n_edges, dtype=np.float64)
        self.edge_cost = np.zeros(self.n_edges, dtype=np.float64)
        self.edge_under_renovation = np.zeros(self.n_edges, dtype=bool)

        # Fill edge data
        for i, (_, link) in enumerate(links.iterrows()):
            self.edge_from[i] = self.node_to_idx[link['init_node']]
            self.edge_to[i] = self.node_to_idx[link['term_node']]

            # Handle different column names for free flow time
            if 'fft' in link:
                self.edge_fft[i] = link['fft']
            elif 'free_flow_time' in link:
                self.edge_fft[i] = link['free_flow_time']

            self.edge_capacity[i] = link['capacity']
            self.edge_cost[i] = self.edge_fft[i]  # Initial cost equals free flow time

        # Create edge lookup dictionary for compatibility
        self.edge_lookup = {}
        for i in range(self.n_edges):
            from_node = self.nodes[self.edge_from[i]]
            to_node = self.nodes[self.edge_to[i]]
            self.edge_lookup[(from_node, to_node)] = i

        # Convert OD pairs to arrays
        self._create_od_arrays(trips)

    def _create_od_arrays(self, trips):
        """Convert OD pairs to efficient array representation."""
        # Group by origin for efficient processing
        od_data = defaultdict(dict)
        for _, trip in trips.iterrows():
            if trip['demand'] > 0:
                origin_idx = self.node_to_idx[trip['init_node']]
                dest_idx = self.node_to_idx[trip['term_node']]
                od_data[origin_idx][dest_idx] = trip['demand']

        # Convert to arrays for faster processing
        self.origins = []
        self.destinations = []
        self.demands = []
        self.od_starts = [0]  # Start indices for each origin

        for origin_idx in sorted(od_data.keys()):
            od_pairs = od_data[origin_idx]
            for dest_idx, demand in od_pairs.items():
                self.origins.append(origin_idx)
                self.destinations.append(dest_idx)
                self.demands.append(demand)
            self.od_starts.append(len(self.origins))

        self.origins = np.array(self.origins, dtype=np.int32)
        self.destinations = np.array(self.destinations, dtype=np.int32)
        self.demands = np.array(self.demands, dtype=np.float64)
        self.od_starts = np.array(self.od_starts, dtype=np.int32)
        self.n_origins = len(self.od_starts) - 1

    @property
    def cost(self):
        if self._cost != np.nan:
            return self._cost
        else:
            raise Exception('tried to access the cost of the network before running it.')

    def reset_flow(self):
        """Reset all edge flows to zero."""
        self.edge_flow.fill(0.0)

    def update_edge_costs(self):
        """Update edge costs using vectorized BPR function."""
        self.edge_cost = BPR_vectorized(self.edge_fft, self.edge_flow, self.edge_capacity)

    def loadAON_array(self):
        """All-or-nothing loading using Numba-optimized Dijkstra."""
        # Initialize flow array
        x_bar = np.zeros(self.n_edges, dtype=np.float64)
        SPTT = 0.0

        # Get unique origins
        unique_origins = np.unique(self.origins)

        for origin_idx in unique_origins:
            # Run Numba Dijkstra from this origin
            distances, predecessors = dijkstra_numba_optimized(
                self.edge_from, self.edge_to, self.edge_cost, origin_idx, self.n_nodes
            )

            # Find OD pairs for this origin
            start_idx = np.searchsorted(self.origins, origin_idx, side='left')
            end_idx = np.searchsorted(self.origins, origin_idx, side='right')

            # Process all destinations for this origin
            for i in range(start_idx, end_idx):
                dest_idx = self.destinations[i]
                demand = self.demands[i]

                if distances[dest_idx] == np.inf or demand <= 0:
                    continue

                # Reconstruct path and add flows using Numba
                path = reconstruct_path_numba_optimized(predecessors, origin_idx, dest_idx)
                if len(path) > 1:
                    add_path_flows_numba(path, demand, self.edge_from, self.edge_to, x_bar)
                elif len(path) == 1 and path[0] == origin_idx == dest_idx:
                    # Origin equals destination, no edges to traverse but still count cost
                    pass

                if len(path) > 0:  # Any valid path
                    SPTT += distances[dest_idx] * demand

        return SPTT, x_bar

    def update_edge_flows_array(self, alpha, direction):
        """Update edge flows using vectorized operations."""
        self.edge_flow = self.edge_flow + alpha * direction

    def get_TSTT_array(self):
        """Calculate Total System Travel Time using vectorized operations."""
        return np.sum(self.edge_cost * self.edge_flow)

    def calculate_conjugate_beta(self, gradient_current, gradient_previous, method='polak-ribiere'):
        """
        Calculate the conjugation parameter beta.

        Args:
            gradient_current: Current gradient (negative of search direction)
            gradient_previous: Previous gradient
            method: 'polak-ribiere' or 'fletcher-reeves'

        Returns:
            beta: Conjugation parameter
        """
        # Polak-Ribiere formula: beta = max(0, g_k^T(g_k - g_{k-1}) / ||g_{k-1}||^2)
        if method == 'polak-ribiere':
            numerator = np.dot(gradient_current, gradient_current - gradient_previous)
            denominator = np.dot(gradient_previous, gradient_previous)
            if denominator < 1e-10:
                return 0.0
            beta = max(0.0, numerator / denominator)

        # Fletcher-Reeves formula: beta = ||g_k||^2 / ||g_{k-1}||^2
        elif method == 'fletcher-reeves':
            numerator = np.dot(gradient_current, gradient_current)
            denominator = np.dot(gradient_previous, gradient_previous)
            if denominator < 1e-10:
                return 0.0
            beta = numerator / denominator

        else:
            raise ValueError(f"Unknown beta method: {method}")

        return beta

    def check_conjugacy_restart(self, direction_current, direction_previous):
        """
        Check if conjugacy should be restarted based on angle between directions.

        Returns:
            True if restart needed, False otherwise
        """
        # Calculate cosine of angle between directions
        norm_current = np.linalg.norm(direction_current)
        norm_previous = np.linalg.norm(direction_previous)

        if norm_current < 1e-10 or norm_previous < 1e-10:
            return True

        cos_angle = np.dot(direction_current, direction_previous) / (norm_current * norm_previous)

        # Restart if directions are not sufficiently aligned
        return cos_angle < self.restart_threshold

    def calculate_CFW_alpha_array(self, direction):
        """
        Calculate optimal step size for conjugate Frank-Wolfe using line search.

        Args:
            direction: Search direction (can be conjugate direction)

        Returns:
            alpha: Optimal step size
        """
        def objective_derivative(alpha):
            """Derivative of objective function with respect to alpha."""
            alpha = max(0.0, min(1.0, alpha))
            tmp_flows = self.edge_flow + alpha * direction
            tmp_costs = BPR_vectorized(self.edge_fft, tmp_flows, self.edge_capacity)
            return np.sum(direction * tmp_costs)

        # Use fsolve to find where derivative equals zero
        sol = fsolve(objective_derivative, np.array([0.5]))

        # Ensure alpha is in valid range
        alpha = np.clip(sol[0], 0.0, 1.0)

        # If alpha is very small, use a minimum step size
        if alpha < 1e-6:
            alpha = 1e-6

        return alpha

    def assignment_loop(self, accuracy=0.001, verbose: bool = None, reset_flows: bool = True):
        """
        Conjugate Frank-Wolfe assignment loop using pure array operations.

        This implementation uses conjugate gradients to improve convergence speed
        compared to standard Frank-Wolfe.
        """
        if verbose is None:
            verbose = self.verbose

        # Check connectivity (simplified - assume connected for performance)
        self.feasible = True
        iteration_number = 1
        gap = np.inf
        assignmentStartTime = time.time()

        if reset_flows:
            self.reset_flow()

        # Initial cost calculation
        self.update_edge_costs()

        # Find x_bar for the first iteration
        _, x_bar = self.loadAON_array()

        # Initialize conjugate direction variables
        direction_previous = np.zeros(self.n_edges, dtype=np.float64)
        gradient_previous = np.zeros(self.n_edges, dtype=np.float64)
        first_iteration = True

        while gap > accuracy:
            # Calculate current direction (AON solution - current flow)
            direction_fw = x_bar - self.edge_flow

            # Calculate gradient (for conjugacy calculation)
            gradient_current = -direction_fw  # Negative because we minimize

            if first_iteration:
                # First iteration: use standard Frank-Wolfe direction
                direction = direction_fw.copy()
                alpha = 1.0  # Standard MSA for first iteration
                first_iteration = False
            else:
                # Calculate conjugation parameter
                beta = self.calculate_conjugate_beta(
                    gradient_current,
                    gradient_previous,
                    method=self.beta_method
                )

                # Calculate conjugate direction: d_k = (y_k - x_k) + beta * d_{k-1}
                direction_conjugate = direction_fw + beta * direction_previous

                # Check if restart is needed (directions too dissimilar)
                if self.check_conjugacy_restart(direction_conjugate, direction_fw):
                    direction = direction_fw.copy()
                    if verbose:
                        print(f"  Conjugacy restarted at iteration {iteration_number}")
                else:
                    direction = direction_conjugate

                # Calculate optimal step size
                alpha = self.calculate_CFW_alpha_array(direction)

            # Ensure alpha is valid
            if alpha <= 0 or np.isnan(alpha):
                alpha = 1e-4

            # Apply flow improvement (vectorized)
            self.update_edge_flows_array(alpha, direction)

            # Ensure non-negative flows
            self.edge_flow = np.maximum(self.edge_flow, 0.0)

            # Compute new travel times (vectorized)
            self.update_edge_costs()

            # Store previous direction and gradient for next iteration
            direction_previous = direction.copy()
            gradient_previous = gradient_current.copy()

            # Relative gap = the convergence measure for user equilibrium.
            #   TSTT = Total System Travel Time at the current flows (sum of cost*flow);
            #   SPTT = Shortest-Path Travel Time = cost of routing all demand on shortest paths
            #          at the current costs (the AON lower bound).
            # At equilibrium every used path is shortest, so TSTT == SPTT and gap -> 0. The loop
            # stops once gap <= accuracy. gap can never be negative (TSTT >= SPTT) — a negative
            # value signals a numerical/path error.
            SPTT, x_bar = self.loadAON_array()
            TSTT = self.get_TSTT_array()
            self._cost = TSTT

            gap = (TSTT / SPTT) - 1
            if gap < 0:
                # Tiny negative gaps are floating-point noise near convergence; gap = 99 forces
                # another iteration. The message is a debug aid, not an error, so keep it verbose-only.
                if verbose:
                    print(f"Error, gap is {gap}. It should never be less than 0.")
                gap = 99
            elif verbose:
                print(f'iter={iteration_number}, alpha={round(alpha, 5)}, gap={round(gap, 5)}')

            iteration_number += 1
            self.iteration_number = iteration_number

            # Check termination conditions
            if iteration_number > self.maxIter:
                if verbose:
                    print("The assignment did not converge to the desired gap and the max number of iterations has been reached")
                    print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
                    print("Current gap:", round(gap, 5))
                return

            if time.time() - assignmentStartTime > self.maxTime:
                if verbose:
                    print("The assignment did not converge to the desired gap and the max time limit has been reached")
                    print("Assignment did ", iteration_number, "iterations")
                    print("Current gap:", round(gap, 5))
                return

        if verbose:
            print("Assignment converged in ", iteration_number, "iterations")
            print("Assignment took", round(time.time() - assignmentStartTime, 5), "seconds")
            print("Current gap:", round(gap, 5))
        return

    def cripple_links(self, links: list[tuple]) -> None:
        """Mark the given links as under construction (see CRIPPLE_* constants).

        ``links`` is a list of ``(init_node, term_node)`` tuples; a single tuple is also accepted.
        Each matched link's free-flow time is multiplied and its capacity reduced so that
        equilibrium routing avoids it, modelling a closed/works road without removing it from the
        graph. Called on a deep copy of the base network (see ``Problem.get_adjusted_network``).
        """
        if isinstance(links[0], int):
            links = [links]

        for link in links:
            self.ongoing_projects.append(link)
            if link in self.edge_lookup:
                edge_idx = self.edge_lookup[link]
                self.edge_under_renovation[edge_idx] = True
                self.edge_fft[edge_idx] *= CRIPPLE_FFT_MULTIPLIER
                self.edge_capacity[edge_idx] *= CRIPPLE_CAPACITY_FACTOR


def plot_traffic_network(network: TrafficNetwork, output_path=None, format='png', add_traffic=True,
                         base_edges=None):
    """
    Plot function for TrafficNetworkCFW that works with array-based data.

    Draws links colored by the congestion ratio (realized / free-flow travel time; green = 1,
    red >= 2) with width proportional to flow, plus a colorbar and a flow-width legend (R1.8).

    Args:
        network: TrafficNetworkCFW instance
        output_path: Optional path to save the plot
        format: Image format for saving
        base_edges: Optional set of directed (from, to) node-id tuples of a *baseline* network.
            When given, links absent from this network but present in the baseline are drawn as
            dashed grey ("removed") and links new to this network are overlaid dashed blue
            ("added"), each with a legend entry — for the topology-variant figures (R1.8).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import networkx as nx
    from matplotlib.lines import Line2D

    # Create a temporary NetworkX graph for plotting (visualization only)
    G = nx.DiGraph()

    # Add nodes with positions from network.node_pos
    for node_id in network.nodes:
        if node_id in network.node_pos['X'] and node_id in network.node_pos['Y']:
            G.add_node(node_id, X=network.node_pos['X'][node_id], Y=network.node_pos['Y'][node_id])

    # Add edges with current flow and cost data from arrays
    for i in range(network.n_edges):
        from_node = network.nodes[network.edge_from[i]]
        to_node = network.nodes[network.edge_to[i]]

        G.add_edge(from_node, to_node,
                   flow=network.edge_flow[i],
                   cost=network.edge_cost[i],
                   fft=network.edge_fft[i],
                   capacity=network.edge_capacity[i],
                   under_renovation=network.edge_under_renovation[i])

    # Set up positions
    pos = {n: (attr['X'], attr['Y']) for n, attr in G.nodes(data=True)}

    # Style configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.figsize": (6, 4),
    })

    # Compute visual attributes
    node_size = 100

    # Edge color: congestion ratio (realized / free-flow travel time), green = 1 -> red >= 2
    # (matches the manuscript caption); dark grey if under renovation.
    ratios = [G[u][v]['cost'] / G[u][v]['fft'] for u, v in G.edges()]
    norm = mcolors.Normalize(vmin=1.0, vmax=2.0)
    colormap = plt.colormaps['RdYlGn_r']
    edge_colors = []
    for i, (u, v) in enumerate(G.edges()):
        if G[u][v].get('under_renovation', False):
            edge_colors.append('darkgrey')
        else:
            edge_colors.append(colormap(norm(min(ratios[i], 2.0))))

    # Edge width: flow
    flows = [G[u][v]['flow'] for u, v in G.edges()]
    max_flow = max(flows) + 1e-6
    widths = [1.0 + 3.0 * (f / max_flow) for f in flows]

    # Start figure
    fig, ax = plt.subplots()

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=widths,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=8,
        node_size=node_size,
    )

    # Topology-difference overlays vs. a baseline network (for the variant figures, R1.8)
    legend_handles, legend_labels = [], []
    if base_edges is not None:
        base_edges = set(base_edges)
        current = set(G.edges())
        removed = base_edges - current
        added = current - base_edges
        for (u, v) in removed:
            if u in pos and v in pos:
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color='0.45', lw=1.4, ls=(0, (4, 3)), zorder=0.5)
        for (u, v) in added:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color='#0033aa', lw=2.2, ls=(0, (4, 3)), zorder=3.5, alpha=0.9)
        if removed:
            legend_handles.append(Line2D([0], [0], color='0.45', lw=1.4, ls=(0, (4, 3))))
            legend_labels.append('link removed (vs. base)')
        if added:
            legend_handles.append(Line2D([0], [0], color='#0033aa', lw=2.2, ls=(0, (4, 3))))
            legend_labels.append('link added (vs. base)')

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color='lightgray',
        edgecolors='black',
        node_size=node_size
    )

    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=4,
        font_color="black"
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.set_axis_off()
    if add_traffic:
        ax.set_title(f'TTT = {network.cost:.2e} (CFW)')

    # Color key (R1.8): congestion-ratio colorbar + flow-width legend
    sm = cm.ScalarMappable(norm=norm, cmap=colormap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02,
                        ticks=[1.0, 1.25, 1.5, 1.75, 2.0])
    cbar.ax.set_yticklabels(['1.0', '1.25', '1.5', '1.75', r'$\geq$2.0'])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('travel-time ratio (realized / free-flow)', fontsize=8)

    for lw, lab in [(1.0, 'low flow'), (2.5, 'medium flow'), (4.0, 'high flow')]:
        legend_handles.append(Line2D([0], [0], color='black', lw=lw))
        legend_labels.append(lab)
    ax.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.01),
              ncol=min(4, len(legend_handles)), fontsize=7, framealpha=0.9,
              borderaxespad=0.2, handlelength=2.2, columnspacing=1.2)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path + '.' + format, format=format, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    from Environments.env.Problem import load_trips_net_data
    from Src.Utils.Utils import INPUT_DIR

    data_dir = str(INPUT_DIR / "Sioux Falls Expanded")  # adjust to inspect a different case study
    net, nodes, trips = load_trips_net_data(data_dir)

    # Create CFW network instance
    network = TrafficNetwork(net, nodes, trips)

    # Test with crippled links
    print("\n" + "=" * 60)
    print("CFW with crippled links:")
    network.verbose = True
    network.assignment_loop(accuracy=0.001)

    # Plot the result
    plot_traffic_network(network, 'cfw_test')
    print("\nPlot saved as 'cfw_test.png'")