"""Construct the Anaheim road-construction case study.

Mirrors `Sioux Falls Expanded/case_constructor.py` (same project-attribute methodology and
settings) but, instead of one project per link, defines the projects as the **80 highest-impact
road corridors** (ranked by mean V/C x total volume) found by running a baseline traffic
assignment.

Pipeline:
  1. Load raw Anaheim net/trips (TNTP-derived) and node coordinates (geojson) from ./raw/.
  2. Run a baseline user-equilibrium assignment to get link flows.
  3. Drop centroid connectors (links touching zone nodes 1..N_ZONES), aggregate directed
     links into undirected roads, and merge contiguous non-branching roads into corridors.
  4. Rank corridors by (mean volume/capacity) x (total volume) and keep the top N_PROJECTS.
     This product targets corridors that are both congested and heavily used, i.e. where
     crippling the link for construction has the largest network travel-delay impact.
  5. Generate per-project attributes (duration, hard due date, k/p decay, cost) and the
     network-level general.csv exactly as the SFE constructor does.
  6. Write general.csv, projects.csv, project_links.csv, net.csv, nodes.csv, trips.csv here.

Run from anywhere:  python "Environments/input/Anaheim/case_constructor.py"
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.optimize as optimize
import scipy.stats as stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from Environments.env.network import TrafficNetwork  # noqa: E402

RAW = os.path.join(HERE, 'raw')

SETTINGS = {
    'time periods per year': 4,
    'avg room per project (years)': 5,
    'avg time per project (years)': 1.2,
    'avg cost per project': 300,
    'avg unused resources': 0.4,
    'construction teams': 9,
    'seed': 1,
}
N_PROJECTS = 80
N_ZONES = 38        # Anaheim: node ids 1..38 are zone centroids
ACCURACY = 0.001    # baseline-equilibrium gap for the congestion ranking


# ----- binomial-decay helpers (identical to the SFE constructor) -----
def find_binomial_p(row):
    n, k = row['hard due date'], row['k_decay']
    func = lambda p: 1 - stats.binom.cdf(k - 1, n, p) - 0.5
    return optimize.bisect(func, 0, 1)


def find_binomial_n(row):
    p, k = row['p_decay'], row['k_decay']
    func = lambda n: 1 - stats.binom.cdf(k - 1, n, p) - 0.5
    for n in range(0, 1000):
        if func(n) > 0:
            return n
    return 999


# ----- raw data loading -----
def load_raw():
    net = pd.read_csv(os.path.join(RAW, 'Anaheim_net.csv'), sep='\t')
    net['init_node'] = net['init_node'].astype(int)
    net['term_node'] = net['term_node'].astype(int)
    trips = pd.read_csv(os.path.join(RAW, 'Anaheim_trips.csv'), sep='\t')
    trips = trips[trips['demand'] > 0].copy()

    with open(os.path.join(RAW, 'anaheim_nodes.geojson')) as f:
        gj = json.load(f)
    coords = {int(ft['properties']['id']): ft['geometry']['coordinates']
              for ft in gj['features']}
    node_ids = sorted(set(net['init_node']) | set(net['term_node'])
                      | set(trips['init_node']) | set(trips['term_node']))
    missing = [n for n in node_ids if n not in coords]
    if missing:
        print(f"WARNING: {len(missing)} nodes lack coordinates; filling (0,0): {missing[:10]}")
    nodes = pd.DataFrame({
        'Node': node_ids,
        'X': [coords.get(n, (0.0, 0.0))[0] for n in node_ids],
        'Y': [coords.get(n, (0.0, 0.0))[1] for n in node_ids],
    })
    return net, nodes, trips


# ----- congestion corridors -----
def build_corridors(net, nodes, trips):
    ntw = TrafficNetwork(net, nodes.set_index('Node'), trips)
    ntw.assignment_loop(accuracy=ACCURACY)
    print(f"baseline assignment: {ntw.iteration_number} iters, TSTT={ntw.cost:.4e}")

    roads = defaultdict(lambda: {'vol': 0.0, 'cap': 0.0, 'links': []})
    for i in range(ntw.n_edges):
        a = int(ntw.nodes[ntw.edge_from[i]])
        b = int(ntw.nodes[ntw.edge_to[i]])
        if a <= N_ZONES or b <= N_ZONES:
            continue  # centroid connector
        key = tuple(sorted((a, b)))
        roads[key]['vol'] += ntw.edge_flow[i]
        roads[key]['cap'] += ntw.edge_capacity[i]
        roads[key]['links'].append((a, b))
    for g in roads.values():
        g['vc'] = g['vol'] / g['cap'] if g['cap'] > 0 else np.inf

    # degrees in the real-road graph
    deg = defaultdict(int)
    incident = defaultdict(list)
    for (u, v) in roads:
        deg[u] += 1
        deg[v] += 1
        incident[u].append((u, v))
        incident[v].append((u, v))

    # union-find: merge roads meeting at a non-branching (degree-2) node into one corridor
    parent = {k: k for k in roads}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for node, rds in incident.items():
        if deg[node] == 2 and len(rds) == 2:
            union(rds[0], rds[1])

    corr = defaultdict(list)
    for k in roads:
        corr[find(k)].append(k)

    rows = []
    for members in corr.values():
        vc_mean = float(np.mean([roads[m]['vc'] for m in members]))
        vol_total = float(sum(roads[m]['vol'] for m in members))
        links = [lk for m in members for lk in roads[m]['links']]
        # Rank by V/C x volume: a corridor's marginal travel-delay impact when crippled grows with
        # both how saturated it is (BPR is convex in V/C) and how many vehicles it carries, so the
        # product targets the corridors where construction timing moves the TTD objective most.
        rows.append({'vc_mean': vc_mean, 'vol_total': vol_total,
                     'score': vc_mean * vol_total, 'n_seg': len(members),
                     'affected links': tuple(links)})
    cdf = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    print(f"roads={len(roads)} -> corridors={len(cdf)}; "
          f"top-{N_PROJECTS} V/Cxvol cutoff={cdf.loc[N_PROJECTS-1, 'score']:.1f} "
          f"(V/C={cdf.loc[N_PROJECTS-1, 'vc_mean']:.3f}, vol={cdf.loc[N_PROJECTS-1, 'vol_total']:.0f})")
    return cdf.head(N_PROJECTS).reset_index(drop=True)


# ----- project attributes (identical methodology to the SFE constructor) -----
def make_projects(affected_links):
    s = SETTINGS
    np.random.seed(s['seed'])
    n = len(affected_links)
    avg_tpp = s['avg time per project (years)'] * s['time periods per year']  # 4.8

    projects = pd.DataFrame(index=pd.RangeIndex(n, name='Project ID'))
    projects['duration'] = np.random.poisson(avg_tpp - 1, n) + 1

    sum_of_work = projects['duration'].sum()
    required_tp = sum_of_work / (s['construction teams'] * (1 - s['avg unused resources']))
    required_years = int(np.ceil(required_tp / s['time periods per year']))
    max_due = required_years * s['time periods per year']
    min_due = s['avg room per project (years)'] * s['time periods per year']

    projects['hard due date'] = (np.random.uniform(min_due, max_due, n).astype(int)
                                 + projects['duration'])
    projects['k_decay'] = np.ceil(
        np.random.uniform(0.1, 0.3, n) * projects['hard due date']).astype(int)
    projects['p_decay'] = projects.apply(find_binomial_p, axis=1)
    projects['hard due date'] = projects.apply(find_binomial_n, axis=1)

    # scalar Poisson draw x duration (matches SFE: cost == const x duration)
    projects['cost'] = np.random.poisson(s['avg cost per project'] / avg_tpp) * projects['duration']

    projects = projects[['p_decay', 'k_decay', 'hard due date', 'duration', 'cost']]

    budget = projects['cost'].sum() / (1 - s['avg unused resources']) / projects['hard due date'].max()
    budget = float(np.ceil(budget / 10) * 10)
    general = {
        'budget': budget,
        'time periods': int(projects['hard due date'].max()),
        'time periods per year': s['time periods per year'],
        'construction teams': s['construction teams'],
        'daily VHT ratio': 0.1,
    }
    project_links = pd.DataFrame({'affected links': affected_links},
                                 index=pd.RangeIndex(n, name='Project ID'))
    return projects, project_links, general


def construct():
    net, nodes, trips = load_raw()
    top = build_corridors(net, nodes, trips)
    projects, project_links, general = make_projects(list(top['affected links']))

    net.to_csv(os.path.join(HERE, 'net.csv'), sep='\t', index=False)
    trips.to_csv(os.path.join(HERE, 'trips.csv'), sep='\t', index=False)
    nodes.to_csv(os.path.join(HERE, 'nodes.csv'), index=False)
    projects.to_csv(os.path.join(HERE, 'projects.csv'))
    project_links.to_csv(os.path.join(HERE, 'project_links.csv'))
    with open(os.path.join(HERE, 'general.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for k, v in general.items():
            w.writerow([k, v])

    print("\nwrote case study to", HERE)
    print("general:", general)
    print(f"projects: {len(projects)}  (duration {projects['duration'].min()}-{projects['duration'].max()}, "
          f"cost/duration={projects['cost'].iloc[0] / projects['duration'].iloc[0]:.0f})")


if __name__ == "__main__":
    construct()
