"""
Buenos Aires Task 4: Advanced Regularization Comparison

This module demonstrates how different regularization techniques (Linear, Ridge, Lasso, ElasticNet)
affect model performance and generalization on Buenos Aires apartment price prediction.

Key concepts:
- Regularization adds a penalty term to prevent overfitting
- Ridge (L2): Shrinks all coefficients uniformly, good for correlated features
- Lasso (L1): Can drive coefficients to zero, performs feature selection
- ElasticNet: Combines L1 and L2, balances Ridge and Lasso strengths
- Alpha parameter controls regularization strength (higher alpha = more regularization)
"""

import warnings
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)


class RegularizationAnalyzer:
    """Comprehensive analysis of regularization techniques for regression."""
    
    def __init__(self, filepath_pattern="data/buenos-aires-real-estate-*.csv"):
        """Initialize analyzer with data loading."""
        self.filepath_pattern = filepath_pattern
        self.X = None
        self.y = None
        self.models = {}
        self.results = {}
        
    def load_and_wrangle_data(self):
        """Load and preprocess Buenos Aires real estate data."""
        def wrangle(filepath):
            df = pd.read_csv(filepath)
            
            # Filter for Capital Federal, apartments, price < $400k
            mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
            mask_apt = df["property_type"] == "apartment"
            mask_price = df["price_aprox_usd"] < 400_000
            df = df[mask_ba & mask_apt & mask_price]
            
            # Remove outliers on surface area
            low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
            df = df[df["surface_covered_in_m2"].between(low, high)]
            
            # Extract coordinates
            df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
            df.drop(columns="lat-lon", inplace=True)
            
            # Extract neighborhood
            df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
            df.drop(columns="place_with_parent_names", inplace=True)
            
            return df
        
        files = sorted(glob(self.filepath_pattern))
        assert len(files) > 0, f"No files found matching {self.filepath_pattern}"
        
        frames = [wrangle(file) for file in files]
        df = pd.concat(frames, ignore_index=True)
        
        self.X = df.drop(columns="price_aprox_usd")
        self.y = df["price_aprox_usd"]
        
        print(f"Data loaded: {len(self.X)} samples, {self.X.shape[1]} features")
        return self
    
    def baseline_analysis(self):
        """Calculate baseline performance using mean prediction."""
        y_mean = self.y.mean()
        y_pred_baseline = np.full_like(self.y, y_mean, dtype=float)
        baseline_mae = mean_absolute_error(self.y, y_pred_baseline)
        baseline_rmse = np.sqrt(mean_squared_error(self.y, y_pred_baseline))
        
        print(f"\nBaseline Analysis")
        print(f"Mean apartment price: ${y_mean:,.0f}")
        print(f"Baseline MAE: ${baseline_mae:,.0f}")
        print(f"Baseline RMSE: ${baseline_rmse:,.0f}")
        
        self.baseline_mae = baseline_mae
        self.baseline_rmse = baseline_rmse
        return self
    
    def build_regularized_models(self):
        """Build and train multiple regularized regression models."""
        # Define models with different alpha values
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
        
        model_configs = {
            'Linear Regression': {
                'model': LinearRegression(),
                'alphas': [None],
            },
            'Ridge (L2)': {
                'model_class': Ridge,
                'alphas': alphas,
            },
            'Lasso (L1)': {
                'model_class': Lasso,
                'alphas': alphas,
                'max_iter': 10000,
            },
            'ElasticNet (L1+L2)': {
                'model_class': ElasticNet,
                'alphas': alphas,
                'max_iter': 10000,
            },
        }
        
        self.models = {}
        
        for model_name, config in model_configs.items():
            if model_name == 'Linear Regression':
                # Linear regression doesn't have alpha
                pipeline = Pipeline([
                    ('encoder', OneHotEncoder(use_cat_names=True)),
                    ('imputer', SimpleImputer()),
                    ('model', config['model']),
                ])
                pipeline.fit(self.X, self.y)
                self.models[f'{model_name} (α=None)'] = {
                    'pipeline': pipeline,
                    'alpha': None,
                }
            else:
                # Models with alpha parameter
                model_class = config['model_class']
                for alpha in config['alphas']:
                    model_kwargs = {'alpha': alpha}
                    if 'max_iter' in config:
                        model_kwargs['max_iter'] = config['max_iter']
                    
                    pipeline = Pipeline([
                        ('encoder', OneHotEncoder(use_cat_names=True)),
                        ('imputer', SimpleImputer()),
                        ('model', model_class(**model_kwargs)),
                    ])
                    pipeline.fit(self.X, self.y)
                    self.models[f'{model_name} (α={alpha})'] = {
                        'pipeline': pipeline,
                        'alpha': alpha,
                    }
        
        print(f"\nBuilt {len(self.models)} regularized models")
        return self
    
    def evaluate_models(self):
        """Evaluate all models on training data and cross-validation."""
        results_list = []
        
        for model_name, model_dict in self.models.items():
            pipeline = model_dict['pipeline']
            y_pred = pipeline.predict(self.X)
            
            mae = mean_absolute_error(self.y, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            r2 = r2_score(self.y, y_pred)
            
            # Cross-validation scores
            cv_mae = -cross_val_score(
                pipeline, self.X, self.y, 
                cv=5, scoring='neg_mean_absolute_error'
            ).mean()
            
            cv_r2 = cross_val_score(
                pipeline, self.X, self.y, 
                cv=5, scoring='r2'
            ).mean()
            
            results_list.append({
                'Model': model_name,
                'Train MAE': mae,
                'Train RMSE': rmse,
                'Train R²': r2,
                'CV MAE': cv_mae,
                'CV R²': cv_r2,
                'Overfitting Gap (MAE)': cv_mae - mae,
            })
        
        self.results = pd.DataFrame(results_list)
        
        # Sort by CV MAE (more realistic estimate)
        self.results = self.results.sort_values('CV MAE')
        
        print("\nModel Evaluation Results")
        print("=" * 80)
        print(self.results.to_string(index=False))
        
        return self
    
    def analyze_coefficient_behavior(self):
        """Analyze how coefficients change with regularization strength."""
        print("\n\nCoefficient Analysis: How Regularization Shrinks Coefficients")
        print("=" * 80)
        
        ridge_models = {k: v for k, v in self.models.items() if 'Ridge' in k}
        lasso_models = {k: v for k, v in self.models.items() if 'Lasso' in k}
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Ridge coefficients
        alphas_ridge = []
        coef_magnitudes_ridge = []
        
        for model_name in sorted(ridge_models.keys()):
            model_dict = ridge_models[model_name]
            if model_dict['alpha'] is not None:
                alphas_ridge.append(model_dict['alpha'])
                pipeline = model_dict['pipeline']
                coefs = pipeline.named_steps['model'].coef_
                coef_magnitudes_ridge.append(np.mean(np.abs(coefs)))
        
        axes[0].plot(alphas_ridge, coef_magnitudes_ridge, 'o-', linewidth=2, markersize=8)
        axes[0].set_xscale('log')
        axes[0].set_xlabel('Alpha (Regularization Strength)', fontsize=11)
        axes[0].set_ylabel('Mean |Coefficient|', fontsize=11)
        axes[0].set_title('Ridge (L2): Gradual Coefficient Shrinkage', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Lasso coefficients and sparsity
        alphas_lasso = []
        coef_magnitudes_lasso = []
        sparsity_lasso = []
        
        for model_name in sorted(lasso_models.keys()):
            model_dict = lasso_models[model_name]
            if model_dict['alpha'] is not None:
                alphas_lasso.append(model_dict['alpha'])
                pipeline = model_dict['pipeline']
                coefs = pipeline.named_steps['model'].coef_
                coef_magnitudes_lasso.append(np.mean(np.abs(coefs)))
                sparsity_lasso.append(np.sum(coefs == 0) / len(coefs))
        
        axes[1].plot(alphas_lasso, coef_magnitudes_lasso, 'o-', linewidth=2, markersize=8, label='Mean |Coef|')
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Alpha (Regularization Strength)', fontsize=11)
        axes[1].set_ylabel('Mean |Coefficient|', fontsize=11)
        axes[1].set_title('Lasso (L1): Feature Elimination', fontsize=12, fontweight='bold')
        
        # Add sparsity on secondary axis
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(alphas_lasso, sparsity_lasso, 's-', color='red', linewidth=2, markersize=8, label='Sparsity')
        ax1_twin.set_ylabel('Fraction of Zero Coefficients', fontsize=11, color='red')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regularization_coefficient_behavior.png', dpi=100, bbox_inches='tight')
        print("\nSaved: regularization_coefficient_behavior.png")
        
        # Print coefficient analysis
        print("\nRidge (L2) Regularization:")
        print("- Coefficients shrink smoothly as alpha increases")
        print("- All features are retained (none eliminated)")
        print("- Good for correlated features (shares the penalty)")
        
        print("\nLasso (L1) Regularization:")
        print(f"- At α=0.01: {np.sum(lasso_models['Lasso (L1) (α=0.01)']['pipeline'].named_steps['model'].coef_ == 0)} features eliminated")
        print(f"- At α=100.0: {np.sum(lasso_models['Lasso (L1) (α=100.0)']['pipeline'].named_steps['model'].coef_ == 0)} features eliminated")
        print("- Coefficients jump to zero (feature selection)")
        print("- Good for sparse solutions (eliminating irrelevant features)")
        
        return self
    
    def compare_with_baseline(self):
        """Compare best regularized model with baseline."""
        print("\n\nComparison with Baseline")
        print("=" * 80)
        
        best_model_name = self.results.iloc[0]['Model']
        best_cv_mae = self.results.iloc[0]['CV MAE']
        
        improvement = (self.baseline_mae - best_cv_mae) / self.baseline_mae * 100
        
        print(f"Best Model: {best_model_name}")
        print(f"Baseline MAE: ${self.baseline_mae:,.0f}")
        print(f"Best Model CV MAE: ${best_cv_mae:,.0f}")
        print(f"Improvement: {improvement:.1f}%")
        
        return self


if __name__ == "__main__":
    analyzer = RegularizationAnalyzer()
    analyzer.load_and_wrangle_data()
    analyzer.baseline_analysis()
    analyzer.build_regularized_models()
    analyzer.evaluate_models()
    analyzer.analyze_coefficient_behavior()
    analyzer.compare_with_baseline()
