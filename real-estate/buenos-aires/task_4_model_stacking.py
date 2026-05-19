"""
Buenos Aires Task 4: Model Stacking and Ensemble Blending

Compare several base regressors and a stacking ensemble for apartment price prediction.
This demonstrates how model stacking can improve generalization by combining complementary models.
"""

import warnings
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.simplefilter(action="ignore", category=FutureWarning)


class ModelStackingAnalyzer:
    """Build base learners, evaluate them, and create a stacking regressor."""

    def __init__(self, filepath_pattern="data/buenos-aires-real-estate-*.csv"):
        self.filepath_pattern = filepath_pattern
        self.X = None
        self.y = None
        self.base_pipelines = {}
        self.results_df = None

    def load_and_wrangle_data(self):
        def wrangle(filepath):
            df = pd.read_csv(filepath)
            mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
            mask_apt = df["property_type"] == "apartment"
            mask_price = df["price_aprox_usd"] < 400_000
            df = df[mask_ba & mask_apt & mask_price]

            low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
            df = df[df["surface_covered_in_m2"].between(low, high)]

            df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
            df.drop(columns="lat-lon", inplace=True)

            df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
            df.drop(columns="place_with_parent_names", inplace=True)

            return df

        files = sorted(glob(self.filepath_pattern))
        assert len(files) > 0, f"No files found matching {self.filepath_pattern}"
        frames = [wrangle(f) for f in files]
        df = pd.concat(frames, ignore_index=True)

        self.X = df.drop(columns="price_aprox_usd")
        self.y = df["price_aprox_usd"]
        print(f"Loaded {len(self.X)} samples")
        return self

    def baseline(self):
        y_mean = self.y.mean()
        mae = mean_absolute_error(self.y, np.full_like(self.y, y_mean, dtype=float))
        print(f"Baseline MAE: {mae:.2f}")
        self.baseline_mae = mae
        return self

    def build_base_learners(self):
        """Create pipelines for base learners (linear + tree-based)."""
        # Linear-family pipeline
        linear_pipe = Pipeline([
            ('encoder', OneHotEncoder(use_cat_names=True)),
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
        ])

        tree_pipe = Pipeline([
            ('encoder', OneHotEncoder(use_cat_names=True)),
            ('imputer', SimpleImputer()),
        ])

        self.base_pipelines = {
            'lr': Pipeline(linear_pipe.steps + [('model', LinearRegression())]),
            'ridge': Pipeline(linear_pipe.steps + [('model', Ridge(alpha=1.0))]),
            'lasso': Pipeline(linear_pipe.steps + [('model', Lasso(alpha=0.1, max_iter=10000))]),
            'rf': Pipeline(tree_pipe.steps + [('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))]),
            'gbr': Pipeline(tree_pipe.steps + [('model', GradientBoostingRegressor(n_estimators=100, random_state=42))]),
        }

        print(f"Built {len(self.base_pipelines)} base learners")
        return self

    def evaluate_base_learners(self, cv=5):
        """Evaluate base learners with cross-validated MAE and training MAE."""
        rows = []
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, pipe in self.base_pipelines.items():
            print(f"Evaluating {name}...")
            cv_mae = -cross_val_score(pipe, self.X, self.y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
            cv_r2 = cross_val_score(pipe, self.X, self.y, cv=kf, scoring='r2', n_jobs=-1).mean()
            pipe.fit(self.X, self.y)
            train_mae = mean_absolute_error(self.y, pipe.predict(self.X))
            train_r2 = r2_score(self.y, pipe.predict(self.X))

            rows.append({
                'Model': name,
                'CV MAE': cv_mae,
                'CV R2': cv_r2,
                'Train MAE': train_mae,
                'Train R2': train_r2,
            })

        self.results_df = pd.DataFrame(rows).sort_values('CV MAE')
        print("\nBase Learner Results:\n", self.results_df.to_string(index=False))
        return self

    def build_and_evaluate_stacking(self, learners_to_stack=None, cv=5):
        """Create a StackingRegressor and evaluate it against base learners."""
        if learners_to_stack is None:
            learners_to_stack = ['ridge', 'rf', 'gbr']

        estimators = [(name, self.base_pipelines[name]) for name in learners_to_stack]

        stacking = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            n_jobs=-1,
            passthrough=False,
        )

        print("Evaluating Stacking Regressor (cv=%d)..." % cv)
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(stacking, self.X, self.y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1).mean()
        cv_r2 = cross_val_score(stacking, self.X, self.y, cv=kf, scoring='r2', n_jobs=-1).mean()

        stacking.fit(self.X, self.y)
        train_mae = mean_absolute_error(self.y, stacking.predict(self.X))
        train_r2 = r2_score(self.y, stacking.predict(self.X))

        stacking_row = {
            'Model': 'stacking_' + '+'.join(learners_to_stack),
            'CV MAE': cv_mae,
            'CV R2': cv_r2,
            'Train MAE': train_mae,
            'Train R2': train_r2,
        }

        self.results_df = pd.concat([self.results_df, pd.DataFrame([stacking_row])], ignore_index=True)
        self.results_df = self.results_df.sort_values('CV MAE')

        print("\nStacking Results Added:\n", pd.DataFrame([stacking_row]).to_string(index=False))

        # Plot comparison
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(self.results_df['Model'], self.results_df['CV MAE'], color='steelblue')
        ax.set_xlabel('CV MAE')
        ax.set_title('Model Comparison: CV MAE (Lower is Better)')
        plt.tight_layout()
        plt.savefig('stacking_model_comparison.png', dpi=100, bbox_inches='tight')
        print("Saved: stacking_model_comparison.png")

        return self


if __name__ == '__main__':
    analyzer = ModelStackingAnalyzer()
    analyzer.load_and_wrangle_data()
    analyzer.baseline()
    analyzer.build_base_learners()
    analyzer.evaluate_base_learners()
    analyzer.build_and_evaluate_stacking()
