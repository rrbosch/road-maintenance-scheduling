"""Work-in-progress emissions simulation -- NOT part of the active optimization loop.

The live objectives are only SL (Tardiness) and TTD (TotalTravelDelay); the Emissions objective
that would consume this was removed in overhaul item 3. The methods below are stubs/placeholders
(they return 0 or ``pass``) kept for a possible future emissions extension.
"""
import numpy as np

class EmisionsSimulation:
    def __init__(self, problem):
        self.problem = problem
        self.results = {}
        self.parts = ['material', 'traffic']

    def get_result(self, x_dict):
        emissions = {}
        # check which other simulations are there
        if 'traffic' in self.parts:
            traffic_emissions = 0
            for ongoing_projects in x_dict['ongoing_projects']:
                traffic_emissions += self.calculate_traffic_emissions(ongoing_projects)
            emissions['traffic'] = traffic_emissions
        if 'logistics' in self.parts:
            logistics_emissions = 0
            for ongoing_projects in x_dict['ongoing_projects']:
                logistics_emissions += self.calculate_logistics_emissions(ongoing_projects)
            emissions['logistics'] = logistics_emissions
        if 'material' in self.parts:
            material_emissions = self.calculate_material_emissions(x_dict)
            emissions['material'] = material_emissions

        total_emissions = 0
        return np.sum(emissions)

    def calculate_traffic_emissions(self, ongoing_projects):
        emissions = 0
        return emissions

    def calculate_material_emissions(self, x_dict):
        pass

    def calculate_logistics_emissions(self, ongoing_projects):
        pass
