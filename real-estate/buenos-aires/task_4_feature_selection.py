"""
Buenos Aires Task 4: Feature Selection Techniques

This module demonstrates multiple feature selection approaches for apartment price prediction:
- SelectKBest (univariate): Scores each feature independently
- Recursive Feature Elimination (RFE): Iteratively removes least important features
- L1-based (Lasso): Features driven to zero during training
- Permutation Importance: Model-agnostic, measures performance drop when shuffled

Each approach has different strengths and identifies different important features.
"""

import warnings
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)


class FeatureSelectionAnalyzer:
    """Comprehensive feature selection analysis using multiple techniques."""
    
    def __init__(self, filepath_pattern="data/buenos-aires-real-estate-*.csv"):
        """Initialize analyzer with data loading."""
        self.filepath_pattern = filepath_pattern
        self.X = None
        self.y = None
        self.feature_names = None
        self.results = {}
        
    def load_and_wrangle_data(self):
        """Load and preprocess Buenos Aires real estate data."""
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
        
        frames = [wrangle(file) for file in files]
        df = pd.concat(frames, ignore_index=True)
        
        self.X = df.drop(columns="price_aprox_usd")
        self.y = df["price_aprox_usd"]
        
        print(f"Data loaded: {len(self.X)} samples")
        print(f"Features before encoding: {list(self.X.columns)}")
        
        return self
    
    def selectkbest_analysis(self, k_values=[5, 10, 15, 20]):
        """Univariate feature selection using SelectKBest with f_regression."""
        print("\n\nSelectKBest (Univariate F-test)")
        print("=" * 80)
        print("How it works:")
        print("- Scores each feature independently based on correlation with target")
        print("- Selects k features with highest scores")
        print("- Fast but ignores feature interactions\n")
        
        # Need to encode features first for univariate scoring
        encoder = OneHotEncoder(use_cat_names=True)
        imputer = SimpleImputer()
        scaler = StandardScaler()
        
        X_encoded = encoder.fit_transform(self.X)
        X_imputed = imputer.fit_transform(X_encoded)
        X_scaled = scaler.fit_transform(X_imputed)
        feature_names = encoder.get_feature_names_out()
        
        # Get SelectKBest scores for all features
        selector_all = SelectKBest(f_regression, k='all')
        selector_all.fit(X_scaled, self.y)
        scores = selector_all.scores_
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Score': scores
        }).sort_values('Score', ascending=False)
        
        print("Top 10 Features by F-Score:")
        print(feature_importance.head(10).to_string(index=False))
        
        self.results['selectkbest'] = feature_importance
        
        # Visualize effect of k
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Top features
        ax1.barh(range(15), feature_importance['Score'].head(15))
        ax1.set_yticks(range(15))
        ax1.set_yticklabels(feature_importance['Feature'].head(15), fontsize=9)
        ax1.set_xlabel('F-Score', fontsize=11)
        ax1.set_title('SelectKBest: Top 15 Features by F-Score', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        
        # Distribution of scores
        ax2.hist(scores, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('F-Score', fontsize=11)
        ax2.set_ylabel('Number of Features', fontsize=11)
        ax2.set_title('Distribution of Feature Importance Scores', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('feature_selection_selectkbest.png', dpi=100, bbox_inches='tight')
        print("\nSaved: feature_selection_selectkbest.png")
        
        return self
    
    def rfe_analysis(self, n_features_range=[5, 10, 15, 20]):
        """Recursive Feature Elimination: iteratively removes least important features."""
        print("\n\nRecursive Feature Elimination (RFE)")
        print("=" * 80)
        print("How it works:")
        print("- Trains model, removes least important feature")
        print("- Repeats until desired number of features remains")
        print("- Accounts for feature interactions (better than univariate)\n")
        
        # Build full pipeline for RFE
        pipeline = Pipeline([
            ('encoder', OneHotEncoder(use_cat_names=True)),
            ('imputer', SimpleImputer()),
            ('ridge', Ridge(alpha=10.0)),  # Use Ridge as estimator
        ])
        
        # Fit to get encoded feature names
        pipeline.fit(self.X, self.y)
        feature_names = pipeline.named_steps['encoder'].get_feature_names_out()
        n_features = len(feature_names)
        
        # RFE with different numbers of features
        cv_scores = {}
        rfe_rankings = None
        
        for n_features_to_select in n_features_range:
            rfe = RFE(
                estimator=Ridge(alpha=10.0),
                n_features_to_select=n_features_to_select,
                step=1
            )
            
            # Create pipeline with RFE
            rfe_pipeline = Pipeline([
                ('encoder', OneHotEncoder(use_cat_names=True)),
                ('imputer', SimpleImputer()),
                ('rfe', rfe),
                ('ridge', Ridge(alpha=10.0)),
            ])
            
            rfe_pipeline.fit(self.X, self.y)
            y_pred = rfe_pipeline.predict(self.X)
            mae = mean_absolute_error(self.y, y_pred)
            cv_scores[n_features_to_select] = mae
            
            if n_features_to_select == 20:  # Save rankings for best subset
                rfe_rankings = rfe.ranking_
                top_features = pd.DataFrame({
                    'Feature': feature_names,
                    'RFE Rank': rfe.ranking_,
                    'Selected': rfe.support_
                }).sort_values('RFE Rank')
        
        print("RFE Rankings (Top 15 features):")
        print(top_features.head(15).to_string(index=False))
        
        self.results['rfe'] = cv_scores
        
        # Visualize performance vs number of features
        fig, ax = plt.subplots(figsize=(10, 5))
        
        features = sorted(cv_scores.keys())
        maes = [cv_scores[f] for f in features]
        
        ax.plot(features, maes, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('Number of Features Selected', fontsize=11)
        ax.set_ylabel('MAE (training)', fontsize=11)
        ax.set_title('RFE: Performance vs Number of Features', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_selection_rfe.png', dpi=100, bbox_inches='tight')
        print("\nSaved: feature_selection_rfe.png")
        
        return self
    
    def lasso_feature_selection(self, alphas=[0.01, 0.1, 1.0, 10.0]):
        """L1-based feature selection: Lasso drives coefficients to zero."""
        print("\n\nLasso (L1) Feature Selection")
        print("=" * 80)
        print("How it works:")
        print("- L1 regularization forces some coefficients to exactly zero")
        print("- Higher alpha = more features eliminated")
        print("- Implicitly selects features during training\n")
        
        results = []
        lasso_models = {}
        
        for alpha in alphas:
            pipeline = Pipeline([
                ('encoder', OneHotEncoder(use_cat_names=True)),
                ('imputer', SimpleImputer()),
                ('lasso', Lasso(alpha=alpha, max_iter=10000)),
            ])
            
            pipeline.fit(self.X, self.y)
            coefs = pipeline.named_steps['lasso'].coef_
            feature_names = pipeline.named_steps['encoder'].get_feature_names_out()
            
            n_features_selected = np.sum(coefs != 0)
            sparsity = 1.0 - (n_features_selected / len(coefs))
            
            y_pred = pipeline.predict(self.X)
            mae = mean_absolute_error(self.y, y_pred)
            
            # Get top features
            importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs,
                'Abs Coefficient': np.abs(coefs)
            }).sort_values('Abs Coefficient', ascending=False)
            
            results.append({
                'Alpha': alpha,
                'Features Selected': n_features_selected,
                'Sparsity': sparsity,
                'MAE': mae,
                'Top Features': importance[importance['Coefficient'] != 0].head(10)
            })
            
            lasso_models[alpha] = pipeline
        
        self.results['lasso'] = results
        
        # Print results
        print("Lasso Feature Elimination at Different Alpha Values:")
        for result in results:
            print(f"\nAlpha = {result['Alpha']}:")
            print(f"  Features selected: {result['Features Selected']}")
            print(f"  Sparsity: {result['Sparsity']:.1%}")
            print(f"  MAE: ${result['MAE']:,.0f}")
            if len(result['Top Features']) > 0:
                print(f"  Top feature: {result['Top Features'].iloc[0]['Feature']} ({result['Top Features'].iloc[0]['Coefficient']:.4f})")
        
        # Visualize sparsity vs alpha
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        alphas_list = [r['Alpha'] for r in results]
        sparsities = [r['Sparsity'] for r in results]
        maes = [r['MAE'] for r in results]
        
        ax1.plot(alphas_list, sparsities, 'o-', linewidth=2, markersize=8, color='coral')
        ax1.set_xscale('log')
        ax1.set_xlabel('Alpha (L1 Regularization)', fontsize=11)
        ax1.set_ylabel('Sparsity (fraction of zero coefficients)', fontsize=11)
        ax1.set_title('Lasso: Feature Elimination vs Regularization Strength', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(alphas_list, maes, 'o-', linewidth=2, markersize=8, color='seagreen')
        ax2.set_xscale('log')
        ax2.set_xlabel('Alpha (L1 Regularization)', fontsize=11)
        ax2.set_ylabel('MAE (training)', fontsize=11)
        ax2.set_title('Lasso: Model Performance vs Alpha', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_selection_lasso.png', dpi=100, bbox_inches='tight')
        print("\nSaved: feature_selection_lasso.png")
        
        return self
    
    def permutation_importance_analysis(self):
        """Model-agnostic feature importance via permutation."""
        print("\n\nPermutation Importance")
        print("=" * 80)
        print("How it works:")
        print("- Trains model on full feature set")
        print("- Randomly shuffles each feature, measures performance drop")
        print("- Model-agnostic (works with any model)")
        print("- Finds real-world impact of features\n")
        
        # Train baseline model
        pipeline = Pipeline([
            ('encoder', OneHotEncoder(use_cat_names=True)),
            ('imputer', SimpleImputer()),
            ('ridge', Ridge(alpha=10.0)),
        ])
        
        pipeline.fit(self.X, self.y)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            pipeline, self.X, self.y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Get feature names
        feature_names = pipeline.named_steps['encoder'].get_feature_names_out()
        
        # Create results dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std,
        }).sort_values('Importance', ascending=False)
        
        print("Top 15 Features by Permutation Importance:")
        print(importance_df.head(15).to_string(index=False))
        
        self.results['permutation'] = importance_df
        
        # Visualize
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_n = 15
        top_features = importance_df.head(top_n)
        
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['Importance'])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['Feature'], fontsize=10)
        ax.set_xlabel('MAE Increase When Shuffled', fontsize=11)
        ax.set_title('Permutation Importance: Top 15 Features', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add error bars
        ax.errorbar(top_features['Importance'], y_pos, 
                   xerr=top_features['Std'], fmt='none', 
                   ecolor='black', alpha=0.3, capsize=3)
        
        plt.tight_layout()
        plt.savefig('feature_selection_permutation.png', dpi=100, bbox_inches='tight')
        print("\nSaved: feature_selection_permutation.png")
        
        return self
    
    def compare_methods(self):
        """Compare insights from different feature selection methods."""
        print("\n\nComparison of Feature Selection Methods")
        print("=" * 80)
        print("\nWhen to use each method:")
        print("\n1. SelectKBest (Univariate)")
        print("   ✓ Fast, good for initial feature screening")
        print("   ✗ Ignores feature interactions")
        print("   Best for: Large feature spaces, quick baseline")
        
        print("\n2. Recursive Feature Elimination (RFE)")
        print("   ✓ Accounts for feature interactions")
        print("   ✓ Can select any subset size")
        print("   ✗ Computationally expensive")
        print("   Best for: When you know exact number of features needed")
        
        print("\n3. Lasso (L1-based)")
        print("   ✓ Automatic feature selection during training")
        print("   ✓ Interpretable coefficients")
        print("   ✓ Fast")
        print("   Best for: Sparse solutions, interpretability")
        
        print("\n4. Permutation Importance")
        print("   ✓ Model-agnostic")
        print("   ✓ Real-world impact on predictions")
        print("   ✓ Handles feature interactions")
        print("   Best for: Understanding trained model behavior")
        
        return self


if __name__ == "__main__":
    analyzer = FeatureSelectionAnalyzer()
    analyzer.load_and_wrangle_data()
    analyzer.selectkbest_analysis()
    analyzer.rfe_analysis()
    analyzer.lasso_feature_selection()
    analyzer.permutation_importance_analysis()
    analyzer.compare_methods()
