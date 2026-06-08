"""Termination criteria selectable by name in ``Config`` (via the registry).

NOTE: these only configure pymoo's own termination object. The live run loop in
``NSGA2.get_res`` ignores it and instead stops at a fixed 24 h wall-clock ``TIME_BUDGET``;
``Config.termination_arg`` is passed here but in practice only names resume-skip files.
"""
from pymoo.termination.default import DefaultMultiObjectiveTermination as DMT
from pymoo.termination.max_gen import MaximumGenerationTermination as MGT


class DefaultMultiObjectiveTermination(DMT):
    """pymoo's default multi-objective termination (tolerance-based), exposed under a local name."""
    pass


class IterationTermination(MGT):
    """Stop after a fixed number of generations (thin wrapper over pymoo's max-generation rule)."""
    def __init__(self, n_gen=200):
        super().__init__(n_max_gen=n_gen)

