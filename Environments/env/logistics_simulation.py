"""Work-in-progress logistics simulation -- NOT part of the active optimization loop.

The live objectives are only SL (Tardiness) and TTD (TotalTravelDelay); the LogisticsCosts
objective that would consume this was removed in overhaul item 3. Route/cost computation here is
incomplete (see the TODOs and the placeholder random cost in ``calculate_logistics``); it is kept
for a possible future logistics-cost extension.
"""
import networkx as nx
import pandas as pd
import numpy as np
from os import path
import pickle

from Src.Utils.Utils import INPUT_DIR

class LogisticsResult:
    def __init__(self, problem, network, feasible: bool, routes: pd.DataFrame):
        self.routes = routes
        self.problem = problem
        self.network = network
        self.feasible = feasible
        self.logistics_dict_path = None
        if self.feasible:
            self.cost = routes['cost'].sum()
        else:
            self.cost = np.inf


class LogisticsSimulation:
    def __init__(self, problem):
        self.logistics_dict: dict[tuple] = {}
        self.logistics_dict_path = None
        self.problem = problem
        self.load_logistics_dict(problem.case_study)

    def load_logistics_dict(self, case_study):
        # load problem dict, or if that doesn't exist, create it
        file_path = str(INPUT_DIR / case_study / 'logistics_dict.pkl')
        self.logistics_dict_path = file_path
        if path.exists(file_path):
            with open(file_path, 'rb') as file:
                temp_dict = pickle.load(file)
            logistics_dict = temp_dict
            print(f"Loaded logistics dict of size {len(logistics_dict)}")
        else:
            # Create the object using Utils.AutoCreateDict()
            logistics_dict = {}  # AutoCreateDict(create_function=self.calculate_logistics)
            print("Created logistics dict.")
        self.logistics_dict = logistics_dict

    def save_logistics_dict(self):
        file_path = self.logistics_dict_path
        logistics_dict = self.logistics_dict  # .to_dict()
        with open(file_path, 'wb') as file:
            pickle.dump(logistics_dict, file)
        print("Stored object in:", file_path)
    # calculate the logistics cost over the entire planning horizon of a single solution (x_dict)
    def get_result(self, x_dict):
        # find if there are any logistics sim results that are missing
        relevant_keys = x_dict['ongoing_projects'] + [tuple()]
        relevant_keys = set(relevant_keys)
        missing_keys = [key for key in relevant_keys if key not in self.logistics_dict.keys()]
        missing_keys = sorted(missing_keys, key=len)
        print(missing_keys)

        # simulate the missing results and store those
        for i, key in enumerate(missing_keys):
            print(f'starting sim {i+1}/{len(missing_keys)}: {key}')
            self.calculate_logistics(key)
            print(f'finished sim {i+1}/{len(missing_keys)}: {key}')

        # then create the evaluation
        LC = []
        baseline_cost = self.logistics_dict[tuple()].cost
        for key in x_dict['ongoing_projects']:
            if not self.logistics_dict[key].feasible:
                return np.inf, False
            else:
                LC_t = self.logistics_dict[key].cost - baseline_cost
                LC.append(LC_t)
        return np.sum(LC_t), True

    # if the requested logistics result does not exist yet, create it
    def calculate_logistics(self, key):
        # get the adjusted logistics network
        network = self.problem.get_adjusted_network(key)

        # check if the network is still connected
        # TODO: We're here. What do we do for the
        strongly_connected = nx.is_strongly_connected(network.G)
        routes = pd.DataFrame(columns=['route', 'cost'])
        if not strongly_connected:
            feasible = False
        else:
            feasible = True
            # go through all jobs, find the materials required, then calculate the route
            for job in key[1]:
                required_resources = job.loc[job, 'resource']
                for resource_id, amount in enumerate(required_resources):
                    if amount > 0:
                        # then find the route
                        # TODO: fix the logistics here
                        route, cost = [], np.random.rand()*100
                        # route, cost = self.find_shortest_route(job, resource_id, network)
                        row_name = pd.Index([(job, resource_id)])
                        data = {'route': [route], 'cost': [cost]}
                        new_row = pd.DataFrame(data=data, index=row_name)
                        routes = pd.concat([routes, new_row])

        # lastly, create the result class that stores the info
        result = LogisticsResult(self.problem, network, feasible, routes)
        return result

    def find_shortest_route(self, job, resource, network, store_alternatives=False):
        origin_nodes = self.problem.input['resources'].loc[resource, 'supply node']
        # origin_nodes = tuple(str(number) for number in origin_nodes)
        destination_nodes = self.problem.input['projects'].loc[job[0], 'affected_links'][0]
        # destination_nodes = tuple(str(number) for number in destination_nodes)

        G = network.networkx_graph
        for e in G.edges():
            a1 = (str(e[0]), str(e[1]))
            a2 = network.linkSet[a1].cost
            G.edges[e[0], e[1]]['weight'] = a2

        if store_alternatives:
            # method where you store all the result and only then find the shortest option
            shortest_routes = {}
            for origin in origin_nodes:
                for destination in destination_nodes:
                    try:
                        shortest_path = nx.shortest_path(G, source=origin, target=destination, weight='weight')
                        shortest_length = nx.shortest_path_length(G, source=origin, target=destination, weight='weight')
                        shortest_routes[(origin, destination)] = {'path': shortest_path, 'length': shortest_length}

                    except nx.NetworkXNoPath:
                        shortest_routes[(origin, destination)] = {'path': None, 'length': float('inf')}

            df = pd.DataFrame(shortest_routes).transpose()
            df['length'] = df['length'].astype(float)
            sri = df['length'].argmin()
            a1 = df['path'].iloc[sri]
            a2 = df['length'].iloc[sri]
            return a1, a2
        else:
            # method where you only remember the shortest option
            best_path = []
            best_length = np.inf
            for origin in origin_nodes:
                for destination in destination_nodes:
                    try:
                        shortest_path = nx.shortest_path(G, source=origin, target=destination, weight='weight')
                        shortest_length = nx.shortest_path_length(G, source=origin, target=destination, weight='weight')
                        if shortest_length < best_length:
                            best_length = shortest_length
                            best_path = shortest_path

                    except nx.NetworkXNoPath:
                        pass

            if best_length < np.inf:
                return best_path, best_length


