# This file contains code for ...
import copy
import csv
import os
import shutil

import pandas as pd

from Environments.env.Problem import Problem_py
from Environments.env.Problem import load_input_data
from Environments.env.network import TrafficNetwork
from Src.Algorithms.Operators.Repair import TestRepair


def road_capacity_modification():
    input_data = load_input_data('Sioux Falls Expanded', ['traffic'])
    total_costs = {}
    multipliers = [.9, 1.1, 100]
    for mult in multipliers:
        net = copy.deepcopy(input_data['net'])
        net['capacity'] = net['capacity'] * mult
        traffic_network = TrafficNetwork(net, input_data['nodes'], input_data['trips'])
        traffic_network.assignment_loop()
        total_costs[mult] = traffic_network.cost * 1e-6

    for mult in multipliers:
        new_case_name = f'Sioux Falls road capacity {mult}'
        new_dir_name = os.path.join('Environments', 'input', new_case_name)
        old_dir_name = os.path.join('Environments', 'input', 'Sioux Falls Expanded')
        os.makedirs(new_dir_name, exist_ok=True)
        for filename in os.listdir(old_dir_name):
            if filename == 'net.csv':
                net = copy.deepcopy(input_data['net'])
                net['capacity'] = net['capacity'] * mult
                dst_path = os.path.join(new_dir_name, filename)
                net.to_csv(dst_path)
            elif filename not in ['traffic_results.db', 'picture.png', 'case_constructor.py']:
                src_path = os.path.join(old_dir_name, filename)
                dst_path = os.path.join(new_dir_name, filename)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)

def scheduling_capacity_modification():
    input_data = load_input_data('Sioux Falls Expanded', ['traffic'])
    env = Problem_py('Sioux Falls Expanded', {'traffic'}, {'SL', 'TTD'})


    def greedy_schedule(env, mult):
           
        durations = env.input['projects']['duration'].values
        costs = env.input['projects']['cost'].values
        deadlines = env.input['projects']['hard due date'].values
        budget_per_timestep = int(env.input['general']['budget'] * mult)
        max_capacity = int(env.input['general']['construction teams'] * mult)

        n_projects = len(durations)
        schedule = []
        time = 0
        budget = 0
        unscheduled = set(range(n_projects))
        active_projects = []  # List of (end_time, project_id)
        ongoing_projects = set()

        # To track finishing times
        finishing_times = [None] * n_projects
        feasible = True
        while unscheduled or active_projects:
            # Update budget and time
            budget += budget_per_timestep

            # Free up finished projects
            active_projects = [(et, pid) for et, pid in active_projects if et > time]
            ongoing_projects = {pid for _, pid in active_projects}

            # Schedule as many as possible, sorted by earliest deadline
            available_projects = sorted(
                list(unscheduled),
                key=lambda pid: deadlines[pid]
            )

            for pid in available_projects:
                if (
                        len(ongoing_projects) < max_capacity and
                        budget >= costs[pid]
                ):
                    # Schedule the project
                    schedule.append((pid, time))
                    unscheduled.remove(pid)
                    budget -= costs[pid]
                    end_time = time + durations[pid]
                    active_projects.append((end_time, pid))
                    ongoing_projects.add(pid)
                    finishing_times[pid] = end_time

            time += 1

            # Avoid infinite loop in pathological case
            if time > deadlines.max():
                feasible = False
                break

        schedule_array = deadlines - durations
        for p, t in schedule:
            schedule_array[p] = t
        if not feasible:
            repair = TestRepair(1000)
            schedule_array = 0
        sum_finish = sum(finishing_times)
        sum_deadline = sum(deadlines)
        usage_ratio = sum_finish / sum_deadline if feasible else False

        return schedule_array, usage_ratio
    
    results_list = []
    for m in range(40):
        mult = 1 - m * 0.01
        schedule, ratio = greedy_schedule(env, mult)
        results_list.append((mult, ratio, schedule))


    total_budget = env.input['general']['budget'] * env.input['general']['time periods']
    total_cost = env.input['projects']['cost'].sum()
    unspent_fraction = 1 - total_cost/total_budget

    total_capacity = env.input['general']['construction teams'] * env.input['general']['time periods']
    total_work = env.input['projects']['duration'].sum()
    unworked_fraction = 1 - total_work/total_capacity

    chosen_m = [0.7, 100]
    for m in chosen_m:
        new_case_name = f'Sioux Falls construction capacity {m}'
        new_dir_name = os.path.join('Environments', 'input', new_case_name)
        old_dir_name = os.path.join('Environments', 'input', 'Sioux Falls Expanded')
        os.makedirs(new_dir_name, exist_ok=True)
        for filename in os.listdir(old_dir_name):
            if filename == 'general.csv':
                general = copy.deepcopy(env.input['general'])
                general['budget'] = int(general['budget'] * m)
                general['construction teams'] = int(general['construction teams'] * m)
                dst_path = os.path.join(new_dir_name, filename)

                with open(dst_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    for key, value in general.items():
                        writer.writerow([key, value])

            elif filename not in ['traffic_results.db', 'picture.png', 'case_constructor.py']:
                src_path = os.path.join(old_dir_name, filename)
                dst_path = os.path.join(new_dir_name, filename)
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)


def network_connectivity_modification():
    input_data = load_input_data('Sioux Falls Expanded', ['traffic'])
    net = copy.deepcopy(input_data['net'])
    traffic_network = TrafficNetwork(net, input_data['nodes'], input_data['trips'])
    traffic_network.assignment_loop()
    old_traffic_cost = traffic_network.cost
    #plot_traffic_network(traffic_network)

    # first make the less connected version
    # links to remove: 10-16, 5-9, 11-12, 14-15
    links_to_remove = [(10, 16), (5, 9), (11, 12), (14, 15)]
    net2 = copy.deepcopy(net)

    def remove_bidirectional_edges(df, edge_list):
        mask = pd.Series([True] * len(df))
        for a, b in edge_list:
            mask &= ~(
                    ((df['init_node'] == a) & (df['term_node'] == b)) |
                    ((df['init_node'] == b) & (df['term_node'] == a))
            )
        return df[mask]

    net2 = remove_bidirectional_edges(net2, links_to_remove)
    net2['capacity'] *= 1.4
    less_connected_network = TrafficNetwork(net2, input_data['nodes'], input_data['trips'])
    less_connected_network.assignment_loop()
    # plot_traffic_network(less_connected_network)

    # then make the more connected version
    links_to_add = [(1, 4), (2, 5), (5, 11), (12, 14), (11, 15), (13, 23), (6, 7)]
    ffts = [5, 5, 4, 6, 7, 5, 5]
    def create_new_edge(df, edge_list, ffts):
        new_edges = []
        for i, edge_pair in enumerate(edge_list):
            average_capacity_in_node_a = net.loc[net['init_node']==edge_pair[0], 'capacity'].mean()
            average_capacity_in_node_b = net.loc[net['init_node']==edge_pair[1], 'capacity'].mean()
            for a, b in [edge_pair, (edge_pair[1], edge_pair[0])]:
                new_edges.append({
                    'init_node': a,
                    'term_node': b,
                    'capacity': (average_capacity_in_node_a + average_capacity_in_node_b)/2,
                    'length': ffts[i],
                    'free_flow_time': ffts[i],
                    'b': 0.15,
                    'power': 4,
                    'speed': 0,
                    'toll': 0,
                    'link_type': 1,
                })
        new_df = pd.DataFrame(new_edges)
        concated_df = pd.concat([df, new_df], ignore_index=True)
        return concated_df

    net3 = copy.deepcopy(net)
    net3 = create_new_edge(net3, links_to_add, ffts)
    net3['capacity'] *= 0.75
    more_connected_network = TrafficNetwork(net3, input_data['nodes'], input_data['trips'])
    more_connected_network.assignment_loop()
    more_connected_traffic_cost = more_connected_network.cost
    ratio = more_connected_traffic_cost / old_traffic_cost
    # plot_traffic_network(more_connected_network)

    # Create both case studies
    env = Problem_py('Sioux Falls Expanded', {'traffic'}, {'SL', 'TTD'})

    cases = [
        {
            'name': 'Sioux Falls Less Connected',
            'net': net2,
            'removed_links': links_to_remove
        },
        {
            'name': 'Sioux Falls More Connected',
            'net': net3,
            'removed_links': []
        }
    ]

    for case in cases:
        case_name = case['name']
        case_net = case['net']
        removed_links = case['removed_links']

        # Create new directory
        new_dir_name = os.path.join('Environments', 'input', case_name)
        old_dir_name = os.path.join('Environments', 'input', 'Sioux Falls Expanded')
        os.makedirs(new_dir_name, exist_ok=True)

        # Copy and modify files
        for filename in os.listdir(old_dir_name):
            src_path = os.path.join(old_dir_name, filename)
            dst_path = os.path.join(new_dir_name, filename)

            if filename == 'net.csv':
                # Save modified network
                case_net.to_csv(dst_path, index=False)

            elif filename == 'projects.csv':
                # Filter projects that affect removed links (for less connected case)
                projects = copy.deepcopy(env.input['projects'])
                if removed_links:
                    # Get projects that affect removed links
                    project_links = env.input['project links']
                    projects_to_remove = []

                    for proj_idx, affected_links_str in project_links['affected links'].items():
                        affected_links = affected_links_str
                        if isinstance(affected_links, tuple):
                            affected_links = [affected_links]
                        # Check if any affected link is in removed links
                        for link_tuple in affected_links:
                            if link_tuple in removed_links or (link_tuple[1], link_tuple[0]) in removed_links:
                                projects_to_remove.append(proj_idx)
                                break

                    # Remove projects that affect removed links
                    projects = projects.drop(projects_to_remove, errors='ignore')

                projects.to_csv(dst_path)

            elif filename == 'project_links.csv':
                # Filter project links for remaining projects
                project_links = copy.deepcopy(env.input['project links'])
                if removed_links:
                    # Remove project links for projects that affect removed links
                    projects_to_remove = []
                    for proj_idx, affected_links_str in project_links['affected links'].items():
                        affected_links = affected_links_str
                        if isinstance(affected_links, tuple):
                            affected_links = [affected_links]
                        for link_tuple in affected_links:
                            if link_tuple in removed_links or (link_tuple[1], link_tuple[0]) in removed_links:
                                projects_to_remove.append(proj_idx)
                                break

                    project_links = project_links.drop(projects_to_remove, errors='ignore')

                project_links.to_csv(dst_path)

            elif filename in ['general.csv', 'nodes.csv', 'trips.csv', 'materials.csv']:
                # Directly copy these files
                if os.path.isfile(src_path):
                    shutil.copy(src_path, dst_path)

            # Skip copying traffic_results.db, picture.png, case_constructor.py, and other files

        print(f"Created case study: {case_name}")

    print("Network connectivity modification completed!")



if __name__ == "__main__":
    # modify existing instance by changing as little as possible from the original instance
    # road_capacity_modification()
    scheduling_capacity_modification()
    # network_connectivity_modification()


