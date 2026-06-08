from itertools import combinations
from typing import List, Tuple, Dict

import numpy as np


class SubsetAdditiveRegressor:
    """
    Custom regressor that builds corrections based on feature subsets.
    Uses 1-layer deep decision stumps: if subset present, add correction.
    """

    def __init__(self, max_subset_size: int = 3, threshold: float = 0.1, verbose: bool = True):
        """
        Parameters:
        -----------
        max_subset_size : int
            Maximum size of feature subsets to consider
        threshold : float
            Minimum improvement in error to accept a correction
        verbose : bool
            Whether to print progress
        """
        self.max_subset_size = max_subset_size
        self.threshold = threshold
        self.verbose = verbose
        self.corrections: List[Tuple[frozenset, float]] = []
        self.base_prediction = 0.0
        self.n_features = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model to training data.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Binary feature matrix (0s and 1s)
        y : np.ndarray, shape (n_samples,)
            Target values (strictly increasing with respect to subsets)
        """
        self.n_features = X.shape[1]
        n_samples = X.shape[0]

        # Start with base prediction (e.g., mean of empty set cases)
        empty_mask = np.all(X == 0, axis=1)
        if np.any(empty_mask):
            self.base_prediction = np.mean(y[empty_mask])
        else:
            self.base_prediction = 0.0

        # Initialize predictions
        current_pred = np.full(n_samples, self.base_prediction)

        # Iterate through subset sizes
        for subset_size in range(1, min(self.max_subset_size + 1, self.n_features + 1)):
            if self.verbose:
                print(f"\nProcessing subsets of size {subset_size}...")

            # Generate all combinations of features of this size
            feature_combinations = list(combinations(range(self.n_features), subset_size))

            # Track improvements for this round
            improvements = []

            for feature_subset in feature_combinations:
                feature_subset = frozenset(feature_subset)

                # Check if this subset is already covered by existing corrections
                if self._is_covered(feature_subset):
                    continue

                # Get mask for samples where ALL features in subset are 1
                mask = self._get_subset_mask(X, feature_subset)

                if not np.any(mask):
                    continue  # No samples have this subset active

                # Calculate current error for samples with this subset
                error = y - current_pred
                error_with_subset = error[mask]

                if len(error_with_subset) == 0:
                    continue

                # Minimal adjustment (conservative estimate)
                minimal_adjustment = np.min(error_with_subset)

                # Skip if adjustment is negligible or negative
                if minimal_adjustment <= 0:
                    continue

                # Calculate relative decline in error if we add this correction
                new_pred = current_pred.copy()
                new_pred[mask] += minimal_adjustment

                old_total_error = np.sum(np.abs(y - current_pred))
                new_total_error = np.sum(np.abs(y - new_pred))
                decline_in_error = old_total_error - new_total_error
                decline_in_error = decline_in_error / (old_total_error + 1e-10)

                improvements.append({
                    'subset': feature_subset,
                    'adjustment': minimal_adjustment,
                    'decline': decline_in_error,
                    'n_affected': np.sum(mask)
                })

            # Sort improvements by decline in error (greedy selection)
            improvements.sort(key=lambda x: x['decline'], reverse=True)

            # Add corrections that meet threshold
            added_count = 0
            for improvement in improvements:
                if improvement['decline'] > self.threshold:
                    # Add this correction
                    self.corrections.append((
                        improvement['subset'],
                        improvement['adjustment']
                    ))

                    # Update predictions
                    mask = self._get_subset_mask(X, improvement['subset'])
                    current_pred[mask] += improvement['adjustment']

                    added_count += 1

                    if self.verbose:
                        print(f"  Added correction for {set(improvement['subset'])}: "
                              f"+{improvement['adjustment']:.4f} "
                              f"(decline: {improvement['decline']:.4f}, "
                              f"affects {improvement['n_affected']} samples)")

            if self.verbose:
                print(f"  Total corrections added for size {subset_size}: {added_count}")

        # Final statistics
        final_pred = self.predict(X)
        final_error = np.mean(np.abs(y - final_pred))

        if self.verbose:
            print(f"\nFitting complete!")
            print(f"Total corrections: {len(self.corrections)}")
            print(f"Base prediction: {self.base_prediction:.4f}")
            print(f"Final MAE: {final_error:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new data.

        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Binary feature matrix

        Returns:
        --------
        predictions : np.ndarray, shape (n_samples,)
        """
        n_samples = X.shape[0]
        predictions = np.full(n_samples, self.base_prediction)

        # Apply each correction
        for feature_subset, adjustment in self.corrections:
            mask = self._get_subset_mask(X, feature_subset)
            predictions[mask] += adjustment

        return predictions

    def _get_subset_mask(self, X: np.ndarray, feature_subset: frozenset) -> np.ndarray:
        """
        Get boolean mask for samples where ALL features in subset are active (=1).
        """
        mask = np.ones(X.shape[0], dtype=bool)
        for feature_idx in feature_subset:
            mask &= (X[:, feature_idx] == 1)
        return mask

    def _is_covered(self, feature_subset: frozenset) -> bool:
        """
        Check if this subset is already covered by existing corrections.
        A subset is covered if it's a superset of any existing correction.
        """
        for existing_subset, _ in self.corrections:
            if existing_subset.issubset(feature_subset) and existing_subset != feature_subset:
                # This subset contains an existing correction, might want to skip
                # or handle differently depending on your needs
                pass
        return False

    def get_corrections_dict(self) -> Dict[frozenset, float]:
        """
        Get corrections as a dictionary for easy lookup.
        """
        return {subset: adjustment for subset, adjustment in self.corrections}

    def summary(self):
        """
        Print a summary of the model.
        """
        print(f"SubsetAdditiveRegressor Summary")
        print(f"=" * 50)
        print(f"Base prediction: {self.base_prediction:.4f}")
        print(f"Number of corrections: {len(self.corrections)}")
        print(f"\nCorrections by subset size:")

        by_size = {}
        for subset, adj in self.corrections:
            size = len(subset)
            if size not in by_size:
                by_size[size] = []
            by_size[size].append((subset, adj))

        for size in sorted(by_size.keys()):
            print(f"\n  Size {size}: {len(by_size[size])} corrections")
            for subset, adj in sorted(by_size[size], key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {set(subset)}: +{adj:.4f}")
            if len(by_size[size]) > 5:
                print(f"    ... and {len(by_size[size]) - 5} more")


# Usage example
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Binary features
    X = np.random.randint(0, 2, size=(n_samples, n_features))

    # Target: sum of features + interactions
    y = np.zeros(n_samples)
    for i in range(n_samples):
        active_features = np.where(X[i] == 1)[0]
        # Base additive effect
        y[i] = len(active_features) * 10
        # Add interaction effects
        if 0 in active_features and 1 in active_features:
            y[i] += 15  # Bonus for features 0 and 1 together
        if 2 in active_features and 3 in active_features and 4 in active_features:
            y[i] += 25  # Bonus for features 2, 3, 4 together

    # Add some noise
    y += np.random.normal(0, 1, n_samples)

    # Fit model
    model = SubsetAdditiveRegressor(max_subset_size=3, threshold=0.5, verbose=True)
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    print(f"\nTest predictions:")
    print(f"MAE: {np.mean(np.abs(y - y_pred)):.4f}")
    print(f"RMSE: {np.sqrt(np.mean((y - y_pred) ** 2)):.4f}")

    # Show summary
    model.summary()


if __name__ == "__main__":
    pass
