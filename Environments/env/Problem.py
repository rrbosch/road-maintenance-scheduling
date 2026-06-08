from __future__ import print_function

import copy
from collections.abc import Iterable
from os import path

import numpy as np
import pandas as pd
from pymoo.core.individual import Individual
from pymoo.core.problem import ElementwiseProblem

import Environments.env.Objectives as ob
import Environments.env.emissions_simulation as es
import Environments.env.logistics_simulation as ls
import Environments.env.network as nt
import Environments.env.traffic_simulation as ts
import Src.Utils.Utils as Utils


class Problem_py(ElementwiseProblem):
    """The scheduling model: decision vector -> objective values + constraint violations.

    A decision vector ``x`` holds one integer **start time per project** (``-1`` = not planned).
    pymoo calls ``_evaluate`` per individual; it checks scheduling feasibility
    (``check_scheduling_constraints``) and, if feasible, computes the objectives via ``get_F``
    (exact ``get_value`` or, for screening, the cheaper ``get_lower_bound``).

    The central intermediate representation is ``get_x_dict(x)``, which expands ``x`` into, per
    time period, the set of ongoing projects and the money spent — feeding both the constraints
    and the objectives. For TTD, each per-period ongoing-project set ("scenario") is realized as a
    crippled copy of the base network via ``get_adjusted_network`` and solved by the traffic sim.

    Construction inputs are loaded from ``Environments/input/<case_study>/`` by ``load_input_data``.
    """
    def __init__(self, case_study, models, objectives, lower_bound=None, lower_bound_quantile=0, traffic_cache_size=200_000):
        # start with loading the requisite data
        input_data = load_input_data(case_study, models)
        self.case_study = case_study
        self.multiprocessing = False
        self.lower_bound = lower_bound
        self.lower_bound_quantile = lower_bound_quantile
        self.traffic_cache_size = traffic_cache_size

        input_data['general']['years'] = int(input_data['general']['time periods'] / input_data['general']['time periods per year'])
        self._input = input_data

        # add simulation entity
        self.set_simulations(models)
        self.set_objectives(objectives)
        self.base_network.assignment_loop()

        # set the number of vars, objectives and constraints
        n_var = self.get_n_vars()
        n_obj = len(self.objectives)
        n_con = self.create_constraints()
        xl, xu = self.create_xl_xu()
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_con, xl=xl, xu=xu, vtype=int)

    def create_xl_xu(self):
        # Per-project start-time bounds: earliest = 0, latest = the last start that still lets the
        # project (of length `duration`) finish on its hard due date.
        xl = np.zeros(self.input['projects'].shape[0])
        xu = self.input['projects']['hard due date'].values - self.input['projects']['duration'].values + 1
        return xl, xu


    @property
    def input(self):
        return self._input

    def get_n_vars(self):
        n_vars = self.input['projects'].shape[0]
        return n_vars

    def create_constraints(self):
        """Build the human-readable constraint registry and return the constraint count.

        Four families of inequality constraints (each written as ``g <= 0`` feasible), plus one
        slot per simulation (currently unused placeholders):
          1. each project starts on/after its release date,
          2. each project finishes on/before its hard due date,
          3. cumulative spending never exceeds the cumulative budget,
          4. at most ``construction teams`` projects run simultaneously in any period.
        The order here must match the rows produced by ``check_scheduling_constraints``.
        """
        self.constraints = []
        # project is started after the release date
        for i in range(self.input['projects'].shape[0]):
            constraint = {
                'constraint nr': 1,
                'project': i,
                'description': f'project {i} on or after after the release date'
            }
            self.constraints.append(constraint)
        # project is finished before the absolute due date
        for i in range(self.input['projects'].shape[0]):
            constraint = {
                'constraint nr': 2,
                'project': i,
                'description': f'project {i} is finished on or before the hard due date'
            }
            self.constraints.append(constraint)
        # we skip the constraint that forces variable z, as it isn't actually in the code

        # budget constraint each year
        for i in range(self.input['general']['time periods']):
            constraint = {
                'constraint nr': 3,
                'time period': i,
                'description': f'the amount of money spent on projects in time period {i} is lower than the budget'
            }
            self.constraints.append(constraint)

        for t in range(self.input['general']['time periods']):
            constraint = {
                'constraint nr': 4,
                'time period': t,
                'description': f"no more than {self.input['general']['construction teams']} projects are worked on simultaneously in time period {t}"
            }
            self.constraints.append(constraint)

        n_con = len(self.constraints) + len(self.sims)
        return n_con

    def seed(self, seed):
        self.seed = seed

    def _evaluate(self, x, out, partial=False, approximate=False, *args, **kwargs):
        """pymoo entry point: fill ``out['F']`` (objectives) and ``out['G']`` (constraints) for one x.

        Infeasible schedules get ``F = inf`` (so they are dominated) but still report their
        constraint violations in ``G``. ``approximate=True`` uses the cheap lower bound instead of
        the exact objective; ``partial=True`` permits ``-1`` (not-planned) entries.
        """
        if x.dtype == 'float64':
            raise Exception('x datatype is a float. Need datatype int for workable differences between populations.')
        if any(x < 0) and not partial:
            raise Exception('Solutions may only contain -1 (i.e. not planned) when doing a partial evaluation.')

        x_dict = self.get_x_dict(x)

        # G holds each constraint in "<= 0 means satisfied" form; any positive entry => infeasible
        G = self.check_scheduling_constraints(x_dict)

        feasible = not bool(np.any(np.array(G) > 0))
        if feasible:
            out["F"] = self.get_F(x_dict, approximate=approximate)
        else:
            out["F"] = [np.inf for _ in self.objectives]

        out["G"] = G

    def get_F(self, x_dict, approximate=False):
        """Objective vector for a feasible schedule (exact, or lower-bound if approximate)."""
        F = []
        for key, objective in self.objectives.items():
            if approximate:
                objective_value = objective.get_lower_bound(self, x_dict)
            else:
                objective_value = objective.get_value(self, x_dict)
            F.append(objective_value)
        return F

    def get_x_dict(self, x):
        """Expand a (possibly partial) start-time vector into the central per-period view.

        Returns a dict with:
          * ``projects``  — the projects table plus each project's ``start`` and ``finish``;
          * ``ongoing_projects`` — for each time period, a ``frozenset`` of the project ids active
            then (these sets are the traffic "scenarios" and the TTD cache keys);
          * ``spending`` — money committed in each time period (a project's full cost lands in its
            start period).
        Accepts partial solutions: projects with ``start == -1`` are simply not planned.
        """
        # Copy the project DataFrame and assign start times
        x_df = self.input['projects'].copy()
        x_df['start'] = x

        # Identify planned projects
        planned_mask = x_df['start'] != -1

        # Only compute finish times for planned projects
        x_df['finish'] = -1
        x_df.loc[planned_mask, 'finish'] = x_df.loc[planned_mask, 'start'] + x_df.loc[planned_mask, 'duration']

        # Vectorize ongoing project detection by pre-filtering only planned projects
        planned_df = x_df[planned_mask]
        project_indices = planned_df.index.values
        starts = planned_df['start'].values
        finishes = planned_df['finish'].values
        T = self.input['general']['time periods']

        # Build ongoing projects per time period using vectorized filtering
        ongoing_projects = [
            frozenset(project_indices[(starts <= t) & (t <= finishes)])
            for t in range(T)
        ]

        # Compute spending per time period (vectorized)
        spending_list = (planned_df.groupby('start')['cost'].sum().reindex(range(T), fill_value=0).tolist())
        x_dict = {
            'projects': x_df,
            'ongoing_projects': ongoing_projects,
            'spending': spending_list,
        }
        return x_dict

    def check_scheduling_constraints(self, x_dict):
        """
        Check whether (partial) schedule is compliant with due dates, budget and capacity. This function accepts impartial x_dict solutions.
        :param x_dict:
        :return:
        """
        projects = x_dict['projects']
        n_projects = len(projects)
        T = self.input['general']['time periods']

        g_list = []

        # Project start after release date (simplified if you have release date info)
        g1 = np.where(projects['start'] >= 0, -projects['start'], 0)
        g_list.append(g1)

        # Project finishes before due date
        valid_finish = projects['finish'] >= 0
        g2 = np.zeros(n_projects)
        g2[valid_finish] = (
                projects.loc[valid_finish, 'finish'].values
                - projects.loc[valid_finish, 'hard due date'].values
        )
        g_list.append(g2)

        # Budget constraint: cumulative spending <= available
        spending = np.array(x_dict['spending'])
        cumulative_spent = np.cumsum(spending)
        available_budget = self.input['general']['budget'] * (np.arange(T) + 1)
        g3 = cumulative_spent - available_budget
        g_list.append(g3)

        # Construction teams constraint
        max_teams = self.input['general']['construction teams']
        g4 = np.array([len(ongoing) - max_teams for ongoing in x_dict['ongoing_projects']])
        g_list.append(g4)

        # Placeholder for simulation constraints (if any)
        g5 = np.zeros(len(self.sims))

        # Combine
        g = np.concatenate(g_list + [g5])
        return g

    def get_adjusted_network(self, project_tuple: frozenset):
        """Deep-copy the base network and cripple every link of the given ongoing projects.

        ``project_tuple`` is one scenario (a frozenset of project ids ongoing in some period). The
        returned crippled network is what the traffic sim solves for that scenario. (Deep-copying
        keeps the base network pristine for the next scenario.)
        """
        new_network: nt.TrafficNetwork = copy.deepcopy(self.base_network)
        for i in project_tuple:
            new_network.ongoing_projects.append(i)
            affected_links = self.input['project links']['affected links'][i]
            new_network.cripple_links(affected_links)
        return new_network

    def set_objectives(self, objectives):
        sims = self.sims.keys()
        objective_classes = {}
        if len(objectives) == 0:
            raise Exception("No objectives were set.")
        if 'SL' in objectives:
            objective_classes['SL'] = ob.Tardiness()
        if 'TTD' in objectives:
            if 'traffic' not in sims:
                raise Exception("traffic simulation is required for TTD calculation, but was not found.")
            objective_classes['TTD'] = ob.TotalTravelDelay(maxsize=self.traffic_cache_size)
            objective_classes['TTD'].add_scenario(frozenset(), self)
        self.objectives = objective_classes

    def set_simulations(self, models):
        self.sims = dict()
        for model in models:
            if model == 'traffic':
                self.base_network = nt.TrafficNetwork(self.input['net'], self.input['nodes'], self.input['trips'])
                self.sims['traffic'] = ts.TrafficSimulation(self)
            elif model == 'logistics':
                self.sims['logistics'] = ls.LogisticsSimulation(self)
            elif model == 'emissions':
                self.sims['emissions'] = es.EmisionsSimulation(self)
            else:
                raise Exception('Invalid model name given.')

    def get_lb(self, pop, decomposed=False):
        """
        Handles either Individuals or decision vectors directly.
        For Individuals, updates their `.data` with lower bound info.
        For raw x vectors, returns a list of dicts with the same info.
        """
        is_individual_input = False
        return_results = []

        # Normalize input
        if isinstance(pop, Individual):
            pop = [pop]
            is_individual_input = True
        elif isinstance(pop, list) and all(isinstance(p, Individual) for p in pop):
            pop = pop
            is_individual_input = True
        elif isinstance(pop, np.ndarray):
            if pop.ndim == 1:
                pop = [pop]
            else:
                pop = [i for i in pop]
        elif isinstance(pop, Iterable) and all(isinstance(p, np.ndarray) for p in pop):  # list of x
            pop = pop
        else:
            raise ValueError("Input must be Individual, list of Individuals, x (np.ndarray), or list of x.")

        for item in pop:
            if is_individual_input:
                x = item.get("X")
            else:
                x = item

            x_dict = self.get_x_dict(x)
            F_lb = []
            F_mi = []

            for key, objective in self.objectives.items():
                objective_lb, objective_missing_info = objective.get_lower_bound(self, x_dict, decomposed=decomposed)
                F_lb.append(objective_lb)
                F_mi.append(objective_missing_info)

            lb_info = Utils.LowerBoundInfo(F_lb=F_lb, missing_info=Utils.flatten_list(F_mi))
            if is_individual_input:
                item.data['lb'] = lb_info
            else:
                return_results.append(lb_info)

        if not is_individual_input:
            return return_results
        else:
            return pop

    def return_constraint_violations(self, g):
        violations = [self.constraints[idx] for idx in np.where(g > 0)[0]]
        return violations



class MultipleMethodsProblem(Problem_py):
    def check_scheduling_constraints(self, x_dict):
        raise NotImplementedError


def load_material_data(data_dir):
    raise NotImplementedError


def load_input_data(case_study, simulations):
    # Anchor to the repo's input dir (via Utils.INPUT_DIR) rather than os.getcwd(), so the case
    # study is found regardless of the working directory the script was launched from.
    data_dir = str(Utils.INPUT_DIR / case_study)

    # determine what data you actually want
    input_data = {'general': load_general_data(data_dir), 'projects': load_project_data(data_dir)}

    for simulation in simulations:
        if simulation == 'traffic':
            if 'trips' not in input_data:
                input_data['net'], input_data['nodes'], input_data['trips'] = load_trips_net_data(data_dir)
                input_data['project links'] = load_project_links_data(data_dir)
        elif simulation == 'logistics':
            if 'trips' not in input_data:
                input_data['trips'], input_data['net'] = load_trips_net_data(data_dir)
                input_data['project links'] = load_project_links_data(data_dir)
            if 'material' not in input_data:
                input_data['material'] = load_material_data(data_dir)
    return input_data


def load_project_links_data(data_dir):
    file_path = path.join(data_dir , "project_links.csv")
    data = pd.read_csv(file_path, delimiter=";", index_col=0)
    if data.shape[1] == 0:
        data = pd.read_csv(file_path, index_col=0)
    data['affected links'] = data['affected links'].apply(Utils.convert_to_tuple)
    data.reset_index(inplace=True)
    return data


def load_general_data(data_dir):
    file_path = path.join(data_dir, 'general.csv')
    data = pd.read_csv(file_path, delimiter=";", header=None, index_col=0)
    if data.shape[1] == 0:
        data = pd.read_csv(file_path, delimiter=",", header=None, index_col=0)
    data2 = data.to_dict()[1]
    data2['time periods'] = int(data2['time periods'])
    data2['construction teams'] = int(data2['construction teams'])
    data2['time periods per year'] = int(data2['time periods per year'])
    return data2


def load_project_data(data_dir):
    file_path = path.join(data_dir , "projects.csv")
    data = pd.read_csv(file_path, delimiter=";", index_col=0)
    if data.shape[1] <= 1:
        data = pd.read_csv(file_path, index_col=0)
    # data['time periods'] = data['time periods'].astype(int)
    data.reset_index(inplace=True)
    return data


def load_trips_net_data(data_dir) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # load trips
    file_path = path.join(data_dir , "trips.csv")
    trips = pd.read_csv(file_path, delimiter="\t")
    if trips.shape[1] <= 1:
        trips = pd.read_csv(file_path, index_col=0)

    # load net
    file_path = path.join(data_dir , "net.csv")
    net = pd.read_csv(file_path, delimiter="\t")
    if net.shape[1] <= 1:
        net = pd.read_csv(file_path)
    if 'Project ID' in net.columns:
        net.set_index('Project ID', inplace=True)

    # load nodes
    file_path = path.join(data_dir , "nodes.csv")
    nodes = pd.read_csv(file_path, delimiter=";", index_col=0)
    if nodes.shape[1] == 0:
        nodes = pd.read_csv(file_path, delimiter=",")
    nodes.set_index('Node', inplace=True)
    return net, nodes, trips


def check_network_sameness(network1, network2):
    diff = {}
    vars_obj1 = vars(network1)
    vars_obj2 = vars(network2)

    for attr, value_obj1 in vars_obj1.items():
        value_obj2 = vars_obj2.get(attr)
        if value_obj1 != value_obj2:
            diff[attr] = (value_obj1, value_obj2)
    if len(diff) > 0:
        print('differences found:')
        print(diff)
    else:
        print('no diffs found.')





if __name__ == "__main__":
    # Random Agent test
    pass
