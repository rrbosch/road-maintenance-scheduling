"""Construct the **SF-12** small instance case study (campaign E1 family / overhaul item 15).

A Sioux Falls instance two projects larger than SF-10 and with a longer horizon, kept within the
regime the branch-and-bound exact solver (`Src/Algorithms/BranchAndBoundSolver.py`) can still prune
its way to the **exact true Pareto front**. The full feasible space is astronomically large, but
B&B's monotone-objective bounds keep it tractable (longer solve times than SF-9/SF-10 are expected
and accepted), extending the exact-ground-truth scaling story SF-8 -> SF-9 -> SF-10 -> SF-12.

Design targets (set by the author):
  * **12 projects**, ranked by **Volume^2 / Capacity** (= mean V/C x total volume — the same product
    score used for SF-9/SF-10 / Anaheim: marginal travel-delay impact when a road is crippled grows
    with both how saturated it is and how much flow it carries).
  * **T = 20 time periods.**
  * **average ~3 projects ongoing simultaneously** -> total work = 3 x 20 = 60 project-periods over
    12 projects -> **mean duration 5.0** periods (Poisson(4)+1, clipped [1,10]).
  * **hard due dates in [15, 20]** periods (so T = max(hard due) = 20; same 5-period deadline window
    as SF-10 relative to the horizon).
  * **construction capacity (teams) = 5** simultaneous projects.
  * **budget effectively disabled** (99999 so the budget constraint never binds; E1 isolates the
    team-capacity + due-date structure).

Methodology is identical to the SF-9/SF-10 constructors (see `Sioux Falls 9/case_constructor.py`):
  1. **Projects = the 12 highest-impact undirected roads** (Volume^2/Capacity from a baseline UE
     assignment). Sioux Falls has *no* centroid connectors (every node is a zone) so no links are
     dropped. Each project closes *both* directions `((a,b),(b,a))`.
  2. **`hard due date` is set directly in [15,20]** and `p_decay` is calibrated so late-start risk
     hits 50% exactly at `start = hard_due` (`Tardiness` uses `n = start`); no `find_binomial_n`
     re-derivation (which would drift the deadline out of [15,20]).

NB this relies on the half-open `[start, start+duration)` ongoing convention fixed in
`Problem.get_x_dict` during item 15 (a project occupies exactly `duration` periods).

The base network (net/nodes/trips) is copied verbatim from `Sioux Falls Expanded`.

Run from anywhere:  python "Environments/input/Sioux Falls 12/case_constructor.py"
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
    'avg time per project (years)': (60 / 12) / 4,  # -> Poisson mean (avg_tpp) = 5.0 periods
    'avg cost per project': 300,
    'avg unused resources': 0.4,                 # only used for the (disabled) cost/budget bookkeeping
    'construction teams': 5,                      # simultaneous-project capacity
    'budget': 99999,                             # effectively disabled (budget constraint never binds)
    'min due': 15,                               # hard due dates drawn uniformly in [min_due, max_due]
    'max due': 20,
    'max duration': 10,                          # clip Poisson tail so xu = hard_due - duration + 1 >= 1
    # seed 6 yields mean duration exactly 5.0 (sum 60 -> avg 3.0 ongoing over T=20), deadlines in
    # [15,20] (max 20). T=20, cap 5 -> capacity binds; far too large to enumerate but still solved
    # exactly by the branch-and-bound solver (longer than SF-10, but accepted).
    'seed': 6,
}
N_PROJECTS = 12
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


# ----- congestion ranking: top-N undirected roads by Volume^2 / Capacity -----
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
        # Rank by Volume^2 / Capacity ( = (vol/cap) x vol = V/C x volume): marginal travel-delay
        # impact when crippled grows with both how saturated the road is (BPR is convex in V/C) and
        # how many vehicles it carries.
        rows.append({'road': (u, v), 'vc': g['vc'], 'vol': g['vol'],
                     'score': g['vol'] ** 2 / g['cap'] if g['cap'] > 0 else np.inf,
                     'affected links': tuple(g['links'])})
    cdf = pd.DataFrame(rows).sort_values('score', ascending=False).reset_index(drop=True)
    print(f"roads={len(roads)} -> top-{N_PROJECTS} vol^2/cap cutoff="
          f"{cdf.loc[N_PROJECTS - 1, 'score']:.1f} "
          f"(V/C={cdf.loc[N_PROJECTS - 1, 'vc']:.3f}, vol={cdf.loc[N_PROJECTS - 1, 'vol']:.0f})")
    return cdf.head(N_PROJECTS).reset_index(drop=True)


# ----- project attributes -----
def make_projects(top):
    s = SETTINGS
    rng = np.random.default_rng(s['seed'])
    n = N_PROJECTS
    avg_tpp = s['avg time per project (years)'] * s['time periods per year']  # 5.0

    # durations ~ Poisson(avg_tpp - 1) + 1  (mean = avg_tpp = 5.0), clipped to [1, max duration]
    duration = np.clip(rng.poisson(avg_tpp - 1, n) + 1, 1, s['max duration'])

    # hard due dates drawn in [min_due, max_due], but never before the project can finish
    hard_due = rng.integers(s['min due'], s['max due'] + 1, n)
    hard_due = np.maximum(hard_due, duration + 1).astype(int)   # guarantee >=1 start option

    # k_decay = ceil(uniform(0.1,0.3) * hard_due), at least 1 and at most hard_due
    k_decay = np.clip(np.ceil(rng.uniform(0.1, 0.3, n) * hard_due).astype(int), 1, hard_due)
    p_decay = np.array([find_binomial_p(int(hard_due[i]), int(k_decay[i])) for i in range(n)])

    # cost kept for completeness (budget is disabled): scalar Poisson x duration, as in SF-9/SF-10/SFE
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
        # T must cover the latest possible finish; pin to max_due so the horizon is the intended 20
        # even if no project happens to draw the latest deadline.
        'time periods': int(max(hard_due.max(), s['max due'])),
        'time periods per year': s['time periods per year'],
        'construction teams': s['construction teams'],
        'daily VHT ratio': 0.1,
    }
    return projects, project_links, general


def report_feasible_space(projects, general, nsamp=400000):
    """Print nominal decision-space size + a Monte-Carlo estimate of the feasible count.

    The nominal space is astronomically large, so the feasible fraction is estimated by uniformly
    sampling start vectors and applying the same constraints the exact/B&B solver uses
    (finish <= hard_due, peak concurrency <= construction teams).
    """
    hard = projects['hard due date'].values.astype(int)
    dur = projects['duration'].values.astype(int)
    xu = hard - dur + 1
    domain = (xu + 1)                               # starts 0..xu inclusive
    nominal = float(np.prod(domain.astype(float)))
    T = general['time periods']
    cap = general['construction teams']

    rng = np.random.default_rng(123)
    S = np.stack([rng.integers(0, int(xu[i]) + 1, nsamp) for i in range(len(dur))], axis=1)
    F = S + dur
    due_ok = (F <= hard).all(axis=1)
    peak = np.zeros(nsamp, int)
    for t in range(T):
        peak = np.maximum(peak, ((S <= t) & (t < F)).sum(axis=1))   # half-open [start, finish)
    frac = float((due_ok & (peak <= cap)).mean())

    print(f"\nper-project start-domain sizes: {domain.tolist()}  -> nominal product = {nominal:,.0f}")
    print(f"time periods T = {T}; construction teams = {cap}; "
          f"total work = {int(dur.sum())} project-periods -> avg ongoing over T = {dur.sum() / T:.2f}")
    print(f"feasible fraction (MC, n={nsamp:,}) = {frac:.4f} -> estimated feasible = {nominal * frac:,.0f}")


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
