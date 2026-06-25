"""Construct the **SF-8** micro-instance case study (campaign E1 / overhaul item 15).

A deliberately tiny Sioux Falls instance whose *full feasible decision space is enumerable* by the
exact Pareto solver (item 16), so PLBE can be measured against the true Pareto front (real HV gap +
measured false-pruning rate).

Design targets (set by the author for E1):
  * **8 projects**, mean **duration 2.25** periods (so total work = 8 x 2.25 = 18 project-periods;
    spread over the ~6-period horizon that is ~3 projects ongoing simultaneously on average).
  * **hard due dates in [4, 6]** periods, so T = max(hard due) = 6 (keeps both the deadlines and the
    scenario horizon consistent and the enumeration well under the ~1e7 exact-solver budget).
  * **construction capacity (teams) = 5** simultaneous projects.
  * **budget effectively disabled** (set to 99999 so the budget constraint never binds; E1 isolates
    the team-capacity + due-date structure).

Methodology mirrors the SFE / Anaheim constructors (same binomial-decay risk model), with two
deliberate departures suited to a micro-instance:
  1. **Projects = the 8 highest-impact undirected roads** (ranked by mean V/C x total volume from a
     baseline UE assignment), exactly the Anaheim ranking — but Sioux Falls has *no* centroid
     connectors (every node is a zone), so no links are dropped. Each project closes *both*
     directions of its road, `((a,b),(b,a))` (more realistic than SFE's one-direction-per-link;
     `cripple_links` accepts the tuple-of-tuples form).
  2. **`hard due date` is set directly in [4,6]** and `p_decay` is calibrated so the late-start risk
     hits 50% exactly at `start = hard_due` (`Tardiness` uses `n = start`). We do NOT run the SFE
     `find_binomial_n` re-derivation, which would drift the deadline out of the target [4,6] range.

`time periods` (T) is set to **max(hard due date)** so the per-period scenario loop and the
team-capacity constraint cover the latest possible project finish (a late-starting project must
remain visible to TTD and the capacity check). With deadlines in [4,6] this gives T <= 6.

The base network (net/nodes/trips) is copied verbatim from `Sioux Falls Expanded` so the file
formats stay identical to what `Problem.load_input_data` expects.

Run from anywhere:  python "Environments/input/Sioux Falls 8/case_constructor.py"
"""
import csv
import os
import shutil
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

# Base network is reused verbatim from the large Sioux Falls instance.
BASE = os.path.join(HERE, '..', 'Sioux Falls Expanded')

SETTINGS = {
    'time periods per year': 4,
    'avg time per project (years)': 2.25 / 4,   # -> Poisson mean (avg_tpp) = 2.25 periods
    'avg cost per project': 300,
    'avg unused resources': 0.4,                 # only used for the (disabled) cost/budget bookkeeping
    'construction teams': 5,                      # simultaneous-project capacity
    'budget': 99999,                             # effectively disabled (budget constraint never binds)
    'min due': 4,                                # hard due dates drawn uniformly in [min_due, max_due]
    'max due': 6,
    'max duration': 4,                           # clip Poisson tail so xu = hard_due - duration + 1 >= 1
    # seed 0 yields mean duration exactly 2.25 (sum 18 -> avg 3.0 ongoing over T=6), deadlines in
    # [4,6], and ~6.3k feasible schedules at cap 5 (8.3% -> capacity meaningfully binds).
    'seed': 0,
}
N_PROJECTS = 8
ACCURACY = 0.001    # baseline-equilibrium gap for the congestion ranking


# ----- binomial-decay calibration -----
def find_binomial_p(n, k):
    """Probability p such that P(X>=k)=0.5 for X~Binomial(n, p) (so risk=50% at start=n=hard_due)."""
    func = lambda p: 1 - stats.binom.cdf(k - 1, n, p) - 0.5
    return optimize.bisect(func, 1e-9, 1 - 1e-9)


# ----- raw data loading (from Sioux Falls Expanded) -----
def load_base():
    net = pd.read_csv(os.path.join(BASE, 'net.csv'))
    if 'Project ID' in net.columns:
        net = net.set_index('Project ID')
    net['init_node'] = net['init_node'].astype(int)
    net['term_node'] = net['term_node'].astype(int)

    nodes = pd.read_csv(os.path.join(BASE, 'nodes.csv'), index_col=0).set_index('Node')
    trips = pd.read_csv(os.path.join(BASE, 'trips.csv'), index_col=0)
    trips = trips[trips['demand'] > 0].copy()
    return net, nodes, trips


# ----- congestion ranking: top-N undirected roads by mean V/C x total volume -----
def build_roads(net, nodes, trips):
    ntw = TrafficNetwork(net, nodes, trips)
    ntw.assignment_loop(accuracy=ACCURACY)
    print(f"baseline assignment: {ntw.iteration_number} iters, TSTT={ntw.cost:.4e}")

    roads = defaultdict(lambda: {'vol': 0.0, 'cap': 0.0, 'links': []})
    for i in range(ntw.n_edges):
        a = int(ntw.nodes[ntw.edge_from[i]])
        b = int(ntw.nodes[ntw.edge_to[i]])
        key = tuple(sorted((a, b)))                 # undirected road
        roads[key]['vol'] += ntw.edge_flow[i]
        roads[key]['cap'] += ntw.edge_capacity[i]
        roads[key]['links'].append((a, b))          # keep each directed link of the road
    for g in roads.values():
        g['vc'] = g['vol'] / g['cap'] if g['cap'] > 0 else np.inf

    rows = []
    for (u, v), g in roads.items():
        # Rank by V/C x volume: marginal travel-delay impact when crippled grows with both how
        # saturated the road is (BPR is convex in V/C) and how many vehicles it carries.
        rows.append({'road': (u, v), 'vc': g['vc'], 'vol': g['vol'],
                     'score': g['vc'] * g['vol'],
                     'affected links': tuple(g['links'])})
    cdf = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    print(f"roads={len(roads)} -> top-{N_PROJECTS} V/Cxvol cutoff="
          f"{cdf.loc[N_PROJECTS - 1, 'score']:.1f} "
          f"(V/C={cdf.loc[N_PROJECTS - 1, 'vc']:.3f}, vol={cdf.loc[N_PROJECTS - 1, 'vol']:.0f})")
    return cdf.head(N_PROJECTS).reset_index(drop=True)


# ----- project attributes -----
def make_projects(top):
    s = SETTINGS
    rng = np.random.default_rng(s['seed'])
    n = N_PROJECTS
    avg_tpp = s['avg time per project (years)'] * s['time periods per year']  # 2.25

    # durations ~ Poisson(avg_tpp - 1) + 1  (mean = avg_tpp = 2.25), clipped to [1, max duration]
    duration = np.clip(rng.poisson(avg_tpp - 1, n) + 1, 1, s['max duration'])

    # hard due dates drawn in [min_due, max_due], but never before the project can finish
    hard_due = rng.integers(s['min due'], s['max due'] + 1, n)
    hard_due = np.maximum(hard_due, duration + 1).astype(int)   # guarantee >=1 start option

    # k_decay = ceil(uniform(0.1,0.3) * hard_due), at least 1 and at most hard_due
    k_decay = np.clip(np.ceil(rng.uniform(0.1, 0.3, n) * hard_due).astype(int), 1, hard_due)
    p_decay = np.array([find_binomial_p(int(hard_due[i]), int(k_decay[i])) for i in range(n)])

    # cost kept for completeness (budget is disabled): scalar Poisson x duration, as in SFE
    cost = rng.poisson(s['avg cost per project'] / avg_tpp) * duration

    projects = pd.DataFrame({
        'p_decay': p_decay,
        'k_decay': k_decay,
        'hard due date': hard_due,
        'duration': duration,
        'cost': cost,
    }, index=pd.RangeIndex(n, name='Project ID'))

    project_links = pd.DataFrame({'affected links': list(top['affected links'])},
                                 index=pd.RangeIndex(n, name='Project ID'))

    general = {
        'budget': s['budget'],
        'time periods': int(hard_due.max()),     # T must cover the latest possible finish
        'time periods per year': s['time periods per year'],
        'construction teams': s['construction teams'],
        'daily VHT ratio': 0.1,
    }
    return projects, project_links, general


def report_feasible_space(projects, general):
    """Print the nominal and (capacity-pruned, sampled) feasible decision-space size."""
    hard = projects['hard due date'].values
    dur = projects['duration'].values
    domain = (hard - dur + 2).astype(int)        # starts 0..(hard-dur+1) inclusive
    nominal = int(np.prod(domain.astype(object)))
    print(f"\nper-project start-domain sizes: {domain.tolist()}  -> nominal product = {nominal:,}")
    # crude average-ongoing estimate from the midpoint placement of each project
    T = general['time periods']
    # With the half-open [start, finish) convention each project occupies exactly `duration`
    # periods, so the total footprint is sum(duration) and avg ongoing = sum(duration)/T.
    print(f"time periods T = {T}; construction teams = {general['construction teams']}; "
          f"total work = {int(dur.sum())} project-periods (footprint = sum(duration)) -> "
          f"avg ongoing if spread over T = {dur.sum() / T:.2f}")


def construct():
    net, nodes, trips = load_base()
    top = build_roads(net, nodes, trips)
    projects, project_links, general = make_projects(top)

    # Copy the base network verbatim (identical formats to what Problem.load_input_data expects).
    for fname in ('net.csv', 'nodes.csv', 'trips.csv'):
        shutil.copyfile(os.path.join(BASE, fname), os.path.join(HERE, fname))

    projects.to_csv(os.path.join(HERE, 'projects.csv'))
    project_links.to_csv(os.path.join(HERE, 'project_links.csv'))
    with open(os.path.join(HERE, 'general.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        for k, v in general.items():
            w.writerow([k, v])

    print("\nwrote case study to", HERE)
    print("general:", general)
    print("projects:\n", projects)
    print("roads (project -> undirected road):")
    for i, road in enumerate(top['road']):
        print(f"  {i}: road {road}  links {project_links['affected links'].iloc[i]}")
    report_feasible_space(projects, general)


if __name__ == "__main__":
    construct()
