"""Name -> class registry for the operators and algorithms that `Config` selects by string.

`Config` stores its choices as plain strings (e.g. ``sampling='WeightedSlackSampling'``,
``evaluator='ApproximateEvaluator'``) so they can live in experiment grids, JSON queues, CLI args,
and the ``config.txt`` log. This module turns those strings into classes with ordinary imports,
replacing the old ``dynamic_load`` reflection trick (which walked the filesystem up to a hardcoded
repo-folder name and depended on ``os.getcwd()``).

To add a new operator/algorithm, import its class and add it to ``_REGISTERED`` below; ``Config``
then accepts its class name as a string with no further wiring.
"""
from Src.Algorithms.Callback import OperatorSuccessCallback, PlottingCallback, StandardCallback
from Src.Algorithms.Evaluators import (
    ApproximateEvaluator,
    LowerBoundEvaluator,
    StandardEvaluator,
)
from Src.Algorithms.Heuristics import (
    IncreasingSlackHeuristic,
    WeightedSlackHeuristic,
    WeightedSlackHeuristicRollout,
)
from Src.Algorithms.Operators.Crossover import (
    CompositeCrossover,
    CustomCrossover,
    NoCrossover,
    NOptCrossover,
    UniformCrossover,
)
from Src.Algorithms.Operators.Mutation import (
    CompositeMutation,
    GeometricMutation,
    MOBasedMutation,
    N_Opt,
    RiskBasedMutation,
    TrafficBasedMutation,
)
from Src.Algorithms.Operators.Repair import TestRepair
from Src.Algorithms.Operators.Sampling import (
    FeasibleRandomSampling,
    IntegerRandomSampling,
    WeightedSlackSampling,
)
from Src.Algorithms.Operators.Termination import (
    DefaultMultiObjectiveTermination,
    IterationTermination,
)

# Every class Config may reference by name. Keyed on the class's own __name__ so the string in a
# config and the class stay in sync automatically.
_REGISTERED = [
    # evaluators
    StandardEvaluator, LowerBoundEvaluator, ApproximateEvaluator,
    # samplers
    IntegerRandomSampling, WeightedSlackSampling, FeasibleRandomSampling,
    # crossovers (CompositeCrossover bundles the rest, but allow selecting any directly)
    CompositeCrossover, NoCrossover, NOptCrossover, CustomCrossover, UniformCrossover,
    # mutations
    CompositeMutation, N_Opt, GeometricMutation, TrafficBasedMutation, RiskBasedMutation, MOBasedMutation,
    # repair
    TestRepair,
    # termination
    IterationTermination, DefaultMultiObjectiveTermination,
    # callbacks
    OperatorSuccessCallback, StandardCallback, PlottingCallback,
    # heuristic algorithms (selected via Config.algo_name)
    WeightedSlackHeuristic, WeightedSlackHeuristicRollout, IncreasingSlackHeuristic,
]

REGISTRY = {cls.__name__: cls for cls in _REGISTERED}


def get(name):
    """Return the registered class for ``name`` (raises a helpful error if unknown)."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"'{name}' is not a registered operator/algorithm. "
            f"Known names: {sorted(REGISTRY)}"
        )
