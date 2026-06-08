import numpy as np
import pandas as pd
import os
import csv
import scipy.stats as stats
import scipy.optimize as optimize


def find_binomial_p(row):
    """
    Find the probability p such that P(X >= k) > 0.5 for a Binomial(n, p) distribution.

    Parameters:
        n (int): The number of trials where P(X >= k) first exceeds 0.5.
        k (int): The threshold for the binomial distribution.

    Returns:
        float: The probability p that satisfies the condition.
    """
    n = row['hard due date']
    k = row['k_decay']
    # Define the function to solve: 1 - CDF(k-1) = 0.5
    func = lambda p: 1 - stats.binom.cdf(k - 1, n, p) - 0.5

    # Use a root-finding method to solve for p
    p_solution = optimize.bisect(func, 0, 1)

    return p_solution

def find_binomial_n(row):
    """
    Find the probability p such that P(X >= k) > 0.5 for a Binomial(n, p) distribution.

    Parameters:
        n (int): The number of trials where P(X >= k) first exceeds 0.5.
        k (int): The threshold for the binomial distribution.

    Returns:
        float: The probability p that satisfies the condition.
    """
    p = row['p_decay']
    k = row['k_decay']
    # Define the function to solve: 1 - CDF(k-1) = 0.5
    func = lambda n: 1 - stats.binom.cdf(k - 1, n, p) - 0.5
    for n in range(0, 1000):
        answer = func(n)
        if answer > 0:
            return n


def case_construction(settings):
    np.random.seed(settings['seed'])
    curr_dir = os.getcwd()

    net_path = curr_dir + '\\raw data\\net.csv'
    net = pd.read_csv(net_path, delimiter="\t")

    nodes_path = curr_dir + '\\raw data\\nodes.csv'
    nodes = pd.read_csv(nodes_path, delimiter=";")

    trips_path = curr_dir + '\\raw data\\trips.csv'
    trips = pd.read_csv(trips_path, delimiter="\t")
    trips = trips[trips['demand'] > 0]

    # create one project per link
    project_columns = ['p_decay', 'k_decay', 'hard due date', 'duration', 'cost']
    projects = pd.DataFrame(data=None, columns=project_columns, index=net.index)
    projects.index.name = "Project ID"

    # create time periods
    avg_time_periods_per_project = settings['avg time per project (years)'] * settings['time periods per year']
    projects['duration'] = np.random.poisson(avg_time_periods_per_project - 1, projects.shape[0]) + 1
    sum_of_work = projects['duration'].sum()
    required_time_periods = sum_of_work / (settings['construction teams'] * (1 - settings['avg unused resources']))
    required_years = int(np.ceil(required_time_periods / settings['time periods per year']))
    max_desired_due_date = required_years * settings['time periods per year']
    min_desired_due_date = settings['avg room per project (years)'] * settings['time periods per year']
    projects['hard due date'] = np.random.uniform(min_desired_due_date, max_desired_due_date, projects.shape[0]).astype(int) + projects['duration']
    projects['k_decay'] = np.ceil(np.random.uniform(low=0.1, high=0.3, size=projects.shape[0]) * projects['hard due date']).astype(int)
    projects['p_decay'] = projects.apply(find_binomial_p, axis=1)
    projects['hard due date'] = projects.apply(find_binomial_n, axis=1)

    # create costs based on project duration and a poisson distribution
    projects['cost'] = np.random.poisson(settings['avg cost per project']/avg_time_periods_per_project) * projects['duration']
    net['init_node'] = net['init_node'].astype(int)
    net['term_node'] = net['term_node'].astype(int)
    projects['affected links'] = net.apply(create_affected_links, axis=1)

    project_links = projects[['affected links']]
    del projects['affected links']

    budget = projects['cost'].sum() / (1 - settings['avg unused resources']) / projects['hard due date'].max()
    budget = np.ceil(budget / 10) * 10
    general = {
        'budget': budget,
        'time periods': projects['hard due date'].max(),
        'time periods per year': settings['time periods per year'],
        'construction teams': settings['construction teams'],
    }

    print("Satisfied with generated case? Press Y to save")
    answer = input()
    if answer == 'Y':
        projects.to_csv('projects.csv')
        net.to_csv('net.csv')
        nodes.to_csv('nodes.csv')
        trips.to_csv('trips.csv')
        project_links.to_csv('project_links.csv')
        # materials.to_csv('materials.csv')
        with open('general.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            for key, value in general.items():
                writer.writerow([key, value])


def create_affected_links(row) -> tuple:
    a = int(row['init_node'])
    b = int(row['term_node'])
    return a, b


if __name__ == "__main__":
    configuration = {
        'time periods per year': 4,
        'avg room per project (years)': 5,
        'avg time per project (years)': 1.2,
        'avg cost per project': 300,
        'avg unused resources': 0.4,
        'construction teams': 9,
        'construction methods mode': 'single',
        'seed': 1,
    }
    case_construction(configuration)