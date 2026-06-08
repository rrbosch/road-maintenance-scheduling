class TrafficSimulation:
    """Thin driver that runs the traffic assignment for "scenarios" and counts the work done.

    A *scenario* is a ``frozenset`` of project ids that are simultaneously under construction in
    some time period; its cost is the equilibrium Total System Travel Time of the network with
    those projects' links crippled. This class only orchestrates: it asks ``Problem`` for the
    crippled network copy and calls its CFW ``assignment_loop`` (see ``network.py``). Caching of
    the results lives in ``TotalTravelDelay.results`` (``Objectives.py``), not here.
    """

    def __init__(self, problem):
        # Equilibrium gap tolerance for each assignment. 0.01 (1%) is loose on purpose: the
        # optimizer runs thousands of assignments and only needs costs accurate enough to rank
        # schedules, so a tighter gap would waste time. (Case-study construction uses a tighter gap.)
        self.accuracy = 0.01
        self.problem = problem
        self.reset_flows = True
        # monotonic count of traffic assignments actually run (used for the 'unique sims' metric
        # and to pace regressor retraining). Counts recomputations after a cache eviction.
        self.n_computed = 0

    def get_multiple_scenarios(self, keys):
        """Compute the network cost for each scenario key (a frozenset of ongoing projects).

        Callers pass only the keys they don't already have cached, so every key here triggers a
        fresh traffic assignment.
        """
        results = {}
        for key in keys:
            new_network = self.problem.get_adjusted_network(key)
            new_network.assignment_loop(accuracy=self.accuracy)
            results[key] = new_network.cost
        self.n_computed += len(results)
        return results


if __name__ == "__main__":
    pass
