"""The two optimization objectives, both minimized.

* ``Tardiness`` (SL) — closed-form, cheap. Penalizes starting projects late via a binomial
  decay risk model (see ``risk_per_project``).
* ``TotalTravelDelay`` (TTD) — expensive. The network-wide extra travel time caused by the
  project schedule; each per-period set of ongoing projects needs a traffic assignment. The
  exact value caches/runs real assignments (``get_value``); the lower bound predicts unseen
  scenario costs with a regressor and reports which costs are still estimated (``get_lower_bound``).
"""
import numpy as np
import xgboost as xgb
from scipy.sparse import csr_matrix
from scipy.stats import binom
from sklearn.model_selection import train_test_split

from Src.Utils.Utils import FIFOCache, frozensets_to_sparse_matrix

# Run +1000 new assignments between regressor retrains (see TotalTravelDelay.update_regressor).
REGRESSOR_RETRAIN_INTERVAL = 1000


class Objective:
    """Interface for an objective. Each returns either an exact value or a lower bound.

    ``get_lower_bound`` additionally returns *missing information*: a list of
    ``(sim, scenario, contribution)`` tuples naming the parts of the bound that are still
    estimated (not exact), so the evaluator can decide whether to refine them.
    """
    def get_value(self, problem, x_dict) -> float:
        raise NotImplementedError

    def get_lower_bound(self, problem, x_dict) -> float:
        raise NotImplementedError

    def get_partial_value(self, problem, x):
        raise NotImplementedError

class Tardiness(Objective):
    """Expected late-start penalty (the "SL" objective), summed over projects.

    Binomial decay model: a project with decay parameters ``(p_decay, k_decay)`` that starts in
    period ``start`` incurs its full ``cost`` with probability ``P(X >= k_decay)`` where
    ``X ~ Binomial(n=start, p=p_decay)`` — i.e. each of the ``start`` periods it was delayed is an
    independent chance ``p_decay`` of a setback, and ``k_decay`` accumulated setbacks trigger the
    penalty. ``P(X >= k) = 1 - CDF(k-1)`` rises with ``start``, so starting earlier is cheaper;
    a not-started project (``start == -1``) contributes zero. The case constructor calibrates
    ``p_decay``/``k_decay`` against each project's hard due date. This is closed-form and cheap.
    """
    def risk_per_project(self, problem, x_dict):
        projects_df = x_dict['projects']

        # Initialize risk array with zeros for all projects (not-started projects stay at 0)
        risk = np.zeros(len(projects_df))

        # Only started projects (start != -1) carry risk
        started_mask = projects_df['start'] != -1

        # P(penalty) = P(X >= k_decay) = 1 - CDF(k_decay - 1) for X ~ Binomial(start, p_decay)
        started_df = projects_df[started_mask]
        probs = 1 - binom.cdf(k=started_df['k_decay'], n=started_df['start'], p=started_df['p_decay'])

        # expected penalty = P(penalty) * cost, placed back at each started project's position
        risk[started_mask] = probs * started_df['cost']

        return risk

    def get_value(self, problem, x_dict) -> float:
        risk = self.risk_per_project(problem, x_dict)
        return risk.sum()

    def get_lower_bound(self, problem, x, decomposed=False):
        # Tardiness is exact and cheap, so its lower bound IS its value and nothing is estimated
        # (empty missing-information list). decomposed=True returns the per-project vector.
        if decomposed:
            return self.risk_per_project(problem, x), []
        else:
            return self.get_value(problem, x), []


class TotalTravelDelay(Objective):
    """Network-wide extra travel time caused by the schedule (the "TTD" objective).

    The TTD of a schedule is the sum over time periods of the equilibrium network cost given the
    projects ongoing in that period. Computing one period's cost is a full traffic assignment, so
    results are cached (``results``) and unseen scenarios are estimated by a regressor for the
    lower-bound path. ``add_scenario`` materializes one scenario's true cost on demand.
    """
    def __init__(self, maxsize=200_000):
        # FIFO-bounded cache of scenario costs. The baseline empty scenario is pinned because it
        # is read directly as the network's base cost (e.g. in get_estimation / get_lower_bound).
        self.results = FIFOCache(maxsize=maxsize, pinned={frozenset()})
        self.regressor = None
        self.n_columns = None
        self.last_update = 0
        self.base_cost = None
        # Held-out surrogate-accuracy rows accumulated at each retrain (drained to surrogate.csv by
        # NSGA2.get_res). Each row: {'n_computed', 'quantile', 'mape', 'pinball_loss'}.
        self.surrogate_log = []

    def add_scenario(self, key, problem):
        """Run the real assignment for one scenario ``key`` and cache its exact cost.

        Used by the evaluator to replace an estimated scenario with its true value.
        """
        scenario_cost = problem.sims['traffic'].get_multiple_scenarios([key])
        self.results.update(scenario_cost)

    def get_value(self, problem, x_dict, per_timeperiod=False):
        if isinstance(x_dict, list):
            ongoing_projects = x_dict
        else:
            ongoing_projects = x_dict['ongoing_projects']
        # Resolve every needed scenario into a local dict. Storing the freshly computed costs can
        # evict old entries from the FIFO cache, so we must not rely on the cache still holding a
        # value between writing and reading it within this call.
        road_keys = set(ongoing_projects)
        costs = {key: self.results[key] for key in road_keys if key in self.results}
        missing_keys = [key for key in road_keys if key not in costs]

        new_costs = problem.sims['traffic'].get_multiple_scenarios(missing_keys)
        costs.update(new_costs)
        self.results.update(new_costs)  # populate the shared cache for future reuse
        # then calculate the TTD
        TTD = [costs[key] for key in ongoing_projects]
        if per_timeperiod:
            return TTD
        else:
            return sum(TTD)

    def get_lower_bound(self, problem, x_dict, decomposed=False):
        """Lower bound on TTD, plus the list of still-estimated scenarios ("missing info").

        For each per-period set of ongoing projects (a "scenario"/op) we use its exact cached cost
        if available, otherwise a regressor *under*-estimate (a low quantile, clamped at the
        baseline cost) — making the sum a valid lower bound on the true TTD. Every estimated
        scenario is recorded in ``missing_info`` as ``('traffic', scenario, contribution)`` so the
        evaluator can later materialize the most impactful one (``add_scenario``) and tighten the
        bound. Identical scenarios across periods are predicted once and reused.
        """
        self.update_regressor(problem)
        missing_info = []
        TTD = []

        # Collapse duplicate per-period scenarios: predict each distinct one once, reuse for all
        # periods where it occurs.
        unique_ops = []
        op_indices = {}  # Maps operation to list of indices where it appears
        for idx, op in enumerate(x_dict['ongoing_projects']):
            if op not in op_indices:
                unique_ops.append(op)
                op_indices[op] = [idx]
            else:
                op_indices[op].append(idx)

        # Try to fetch all results from cache
        cached_results = {}
        missing_ops = []
        for op in unique_ops:
            try:
                cached_results[op] = self.results[op]
            except KeyError:
                missing_ops.append(op)

        # Batch estimation for all missing operations
        if missing_ops:
            if self.regressor is None:
                # No regressor, use baseline
                baseline = self.results[frozenset()]
                for op in missing_ops:
                    cached_results[op] = baseline
                    missing_info.append(('traffic', op, baseline))
            else:
                # Batch convert to sparse matrix
                estimations = self.get_estimation(missing_ops)
                for op, est in zip(missing_ops, estimations):
                    cached_results[op] = est
                    missing_info.append(('traffic', op, est))

        # Build TTD array using cached results
        TTD = np.array([cached_results[op] for op in x_dict['ongoing_projects']])

        if decomposed:
            return TTD, missing_info
        else:
            lower_bound = TTD.sum()
            return lower_bound, missing_info

    def get_partial_value(self, problem, x):
        """Exact TTD for a (possibly partial) start-time vector ``x`` (NaN = not planned).

        Expands ``x`` into per-period frozensets of ongoing projects and evaluates them exactly.
        """
        ongoing_projects = [[] for _ in range(problem.input['general']['time periods'])]
        for i, start in enumerate(x):
            if not np.isnan(start):
                end = start + problem.input['projects'].loc[i, 'time periods']
                for t in range(start, end):
                    ongoing_projects[t].append(i)
        ongoing_projects = [frozenset(i) for i in ongoing_projects]
        return self.get_value(problem, ongoing_projects)

    def update_regressor(self, problem):
        """Lazily warm the cache and (re)fit the TTD lower-bound regressor.

        On first call it seeds the cache with the baseline (empty) scenario plus every
        single-project scenario, giving the regressor a minimal training set. Thereafter it
        refits every ``REGRESSOR_RETRAIN_INTERVAL`` new assignments, training on whatever
        scenarios are currently cached. The regressor variant is chosen by ``problem.lower_bound``.
        """
        if self.n_columns is None:
            self.n_columns = problem.n_var
        if len(self.results) == 0:
            # seed: baseline empty set + one scenario per single project
            a = [[i] for i in problem.input['projects'].index.values] + [[]]
            b = [frozenset(i) for i in a]
            self.results.update(problem.sims['traffic'].get_multiple_scenarios(b))
        # Pace retraining by total simulations run rather than cache size: once the FIFO cache
        # is full its length stops growing, so a len()-based trigger would never fire again.
        n_computed = problem.sims['traffic'].n_computed
        if n_computed - self.last_update > REGRESSOR_RETRAIN_INTERVAL:
            self.last_update = n_computed
            x = frozensets_to_sparse_matrix(list(self.results.keys()), self.n_columns)
            y = np.array(list(self.results.values()))
            gq = problem.lower_bound_quantile
            # default held-out set = full data (in-sample) unless a variant carves a real split
            X_val, y_val = x, y
            if problem.lower_bound == "XGBoost":
                self.regressor = xgb.XGBRegressor(n_estimators=1000, max_depth=8, grow_policy='lossguide', objective="reg:quantileerror", quantile_alpha=gq,
                                                  learning_rate=0.1, reg_lambda=0.1, tree_method="hist", early_stopping_rounds=10)
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
                self.regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                X_val, y_val = X_test, y_test
            elif problem.lower_bound == "XGBoost2":
                # from MSE objective, in-sample prediction accuracy, leaf-budget 4000
                max_leaves = 14
                n_estimators = len(self.results) // (max_leaves * 100)
                print(f"TRAINING XGBOOST. LEAF BUDGET = {max_leaves * n_estimators}")
                # self.regressor = xgb.XGBRegressor(n_estimators=n_estimators, max_leaves=max_leaves, grow_policy='lossguide', reg_lambda=0, learning_rate=0.5, tree_method="hist")
                # self.regressor.fit(x, y, verbose=False)
                params = {
                    'max_leaves': 14,
                    'grow_policy': 'lossguide',
                    'reg_lambda': 0,
                    'learning_rate': 0.5,
                    'tree_method': 'hist',
                    'objective': "reg:squarederror"
                }
                dtrain = xgb.DMatrix(x, label=y)
                self.regressor = xgb.train(params=params, dtrain=dtrain, num_boost_round=n_estimators, verbose_eval=False)
            elif problem.lower_bound == "XGBoostMSE":
                self.regressor = xgb.XGBRegressor(n_estimators=10000, max_depth=8, grow_policy='lossguide', learning_rate=0.1, early_stopping_rounds=10)
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
                self.regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                X_val, y_val = X_test, y_test
            elif problem.lower_bound == "Heuristic":
                self.regressor = SubsetMaxRegressor()
                self.regressor.fit(x, y)
            else:
                raise NotImplementedError

            # record held-out surrogate accuracy for the E2 learning curve (telemetry only)
            self._log_surrogate_accuracy(X_val, y_val, n_computed, gq)

    def _log_surrogate_accuracy(self, X_val, y_val, n_computed, quantile):
        """Append a (MAPE, pinball-loss) row evaluated on ``X_val``/``y_val`` for the surrogate log."""
        try:
            preds = self._predict_raw(X_val)
            y_val = np.asarray(y_val, dtype=float)
            resid = y_val - preds
            nonzero = np.abs(y_val) > 0
            mape = float(np.mean(np.abs(resid[nonzero]) / np.abs(y_val[nonzero]))) if nonzero.any() else float('nan')
            # pinball (quantile) loss at the surrogate's quantile
            pinball = float(np.mean(np.maximum(quantile * resid, (quantile - 1.0) * resid)))
        except Exception:
            mape, pinball = float('nan'), float('nan')
        self.surrogate_log.append({'n_computed': n_computed, 'quantile': quantile,
                                   'mape': mape, 'pinball_loss': pinball})

    def _predict_raw(self, X):
        """Predict with whichever regressor variant is active (Booster needs a DMatrix)."""
        if isinstance(self.regressor, xgb.Booster):
            return self.regressor.predict(xgb.DMatrix(X))
        return self.regressor.predict(X)

    def get_estimation(self, op):
        """Predicted cost for one or more scenarios ``op`` (list of frozensets).

        Falls back to the baseline cost before any regressor exists. Predictions are clamped to be
        at least the baseline (empty-scenario) cost: adding construction can only increase travel
        time, so this keeps the estimate a sane lower bound.
        """
        if self.regressor is None:
            return self.results[frozenset()]
        else:
            x_test = frozensets_to_sparse_matrix(op, self.n_columns)
            result2 = self.regressor.predict(x_test)
            result2 = np.maximum(result2, self.results[frozenset()])
            return result2


class SubsetMaxRegressor:
    """Monotone non-ML lower-bound predictor (``lower_bound='Heuristic'``).

    Predicts a scenario's cost as the max observed cost among all *training subsets* of it. Because
    crippling more links never reduces travel time, any subset's true cost is a valid lower bound
    for a superset — so the max over subsets is the tightest such bound from the data alone.
    """
    def __init__(self):
        self.train_sets = []
        self.y_train = []

    def fit(self, X_train, y_train):
        """
        X_train: binary matrix (dense or sparse), shape (n_samples, n_features)
        y_train: array-like of positive floats, shape (n_samples,)
        """
        if not isinstance(X_train, csr_matrix):
            X_train = csr_matrix(X_train)

        self.train_sets = [set(X_train[i].indices) for i in range(X_train.shape[0])]
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        """
        X_test: binary matrix (dense or sparse), shape (n_samples, n_features)
        Returns: array of predicted max y_train values, shape (n_samples,)
        """
        if not isinstance(X_test, csr_matrix):
            X_test = csr_matrix(X_test)

        predictions = []
        for i in range(X_test.shape[0]):
            test_set = set(X_test[i].indices)
            max_val = 0.0
            for train_set, y in zip(self.train_sets, self.y_train):
                if train_set.issubset(test_set):
                    max_val = max(max_val, y)
            predictions.append(max_val)
        return np.array(predictions)

