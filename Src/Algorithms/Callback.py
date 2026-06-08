import math

import matplotlib.pyplot as plt
from pymoo.core.callback import Callback


class OperatorSuccessCallback(Callback):
    """After each generation, let the composite operators learn from this generation's offspring.

    Forwards the offspring to ``crossover.notify_successes`` / ``mutation.notify_successes`` so the
    adaptive operators can update their per-operator success rates (which then bias selection in
    the next generation). Skips the first call, before any offspring exist.
    """
    def __init__(self):
        super().__init__()
        self.first_iteration = True


    def notify(self, algorithm):
        # Need a population AND offspring; skip the very first generation (no offspring yet)
        if not hasattr(algorithm, 'pop') or not hasattr(algorithm, 'off') or self.first_iteration:
            self.first_iteration = False
            return

        offspring = algorithm.off

        # Call success tracker on both operators
        crossover = algorithm.mating.crossover
        if hasattr(crossover, "notify_successes"):
            crossover.notify_successes(offspring)

        mutation = algorithm.mating.mutation
        if hasattr(mutation, "notify_successes"):
            mutation.notify_successes(offspring)


class StandardCallback(Callback):
    """No-op callback (pymoo default behavior); use when no per-generation hook is wanted."""
    pass


class PlottingCallback(Callback):
    """Debug callback: live-scatter the current Pareto front in objective space each generation."""
    def __init__(self):
        super().__init__()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("NSGA-II Population")
        self.ax.set_xlabel("Objective 1")
        self.ax.set_ylabel("Objective 2")
        plt.ion()  # Enable interactive mode for live updates

    def notify(self, algorithm):
        # Get the current population's objectives
        F = algorithm.opt.get("F")

        # Clear the previous scatter plot
        self.ax.clear()
        self.ax.set_title("NSGA-II Population")
        self.ax.set_xlabel("Objective 1")
        self.ax.set_ylabel("Objective 2")

        # Scatter plot of the population (objective space)
        self.ax.scatter(F[:, 0], F[:, 1], c='blue')
        self.ax.set_xlim(left=0)
        self.ax.set_ylim(bottom=0)
        plt.draw()  # Update the plot
        # sleep(1)
        plt.pause(1)  # Pause briefly to allow plot to update

def round_up_to_two_sig_digits(value):
    """Round ``value`` up to two significant digits (used for plot axis limits)."""
    if value == 0:
        return 0
    else:
        magnitude = math.floor(math.log10(value))
        factor = 10**(magnitude - 1)
        return math.ceil(value / factor) * factor