import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import plot_tree


class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis for earthquake damage prediction.
    
    Different methods reveal different insights:
    - Tree-based importance: How often features split
    - Permutation importance: How model degrades when feature is shuffled
    - Coefficient analysis: Direction and magnitude of feature effects
    """
    
    def __init__(self, model, X_test, y_test, feature_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
    
    def get_tree_importance(self):
        """
        Extract feature importance from Decision Tree.
        
        Interpretation:
        - Measures how much each feature decreases impurity (Gini/entropy)
        - Higher value = feature is more important for splits
        - Biased towards high-cardinality features
        """
        # Extract Decision Tree from pipeline
        dt_model = self.model.named_steps['dt']
        
        importance = pd.Series(
            dt_model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)
        
        return importance
    
    def get_permutation_importance(self, n_repeats=10):
        """
        Calculate permutation importance.
        
        How it works:
        1. Measure baseline model performance
        2. Shuffle each feature randomly
        3. Measure performance drop
        4. Feature importance = performance drop
        
        Advantages:
        - Model-agnostic (works with any model)
        - Unbiased across feature types
        - Tells which features actually impact predictions
        """
        perm_importance = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importance = pd.DataFrame({
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }, index=self.feature_names).sort_values('importance', ascending=False)
        
        return importance
    
    def plot_feature_importance_comparison(self, tree_imp=None, perm_imp=None):
        """
        Compare different importance measures.
        """
        fig, axes = plt.subplots(1, 2 if tree_imp is not None and perm_imp is not None else 1,
                                figsize=(14, 6))
        
        if tree_imp is not None:
            ax = axes[0] if perm_imp is not None else axes
            tree_imp.head(10).sort_values().plot(kind='barh', ax=ax, color='steelblue')
            ax.set_title('Tree-Based Feature Importance')
            ax.set_xlabel('Importance Score')
        
        if perm_imp is not None:
            ax = axes[1]
            perm_imp['importance'].head(10).sort_values().plot(kind='barh', ax=ax, color='coral')
            ax.errorbar(perm_imp['importance'].head(10).sort_values().values,
                       range(len(perm_imp.head(10))),
                       xerr=perm_imp['std'].head(10).sort_values().values,
                       fmt='none', ecolor='black', capsize=3)
            ax.set_title('Permutation Feature Importance')
            ax.set_xlabel('Importance Score')
        
        plt.tight_layout()
        return fig
    
    def analyze_feature_groups(self):
        """
        Group features by domain and analyze importance.
        """
        feature_groups = {
            'Age': ['age_building'],
            'Size': ['plinth_area_sq_ft', 'height_ft_pre_eq'],
            'Foundation': ['foundation_type', 'land_surface_condition'],
            'Structure': ['roof_type', 'ground_floor_type', 'other_floor_type', 'superstructure'],
            'Configuration': ['position', 'plan_configuration']
        }
        
        # Get tree importance
        tree_imp = self.get_tree_importance()
        
        group_importance = {}
        for group_name, features in feature_groups.items():
            total_importance = sum(tree_imp.get(f, 0) for f in features)
            group_importance[group_name] = total_importance
        
        return pd.Series(group_importance).sort_values(ascending=False)
    
    def plot_feature_groups(self):
        """
        Visualize importance by feature groups.
        """
        group_imp = self.analyze_feature_groups()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        group_imp.sort_values().plot(kind='barh', ax=ax, color='teal')
        ax.set_title('Feature Group Importance\n(Building Damage Prediction)')
        ax.set_xlabel('Cumulative Importance')
        ax.grid(axis='x', alpha=0.3)
        
        return fig
    
    def interpret_top_features(self, n_top=5):
        """
        Provide business interpretation of top features.
        """
        tree_imp = self.get_tree_importance()
        top_features = tree_imp.head(n_top)
        
        interpretations = {
            'age_building': 'Older buildings are more vulnerable to earthquake damage',
            'plinth_area_sq_ft': 'Larger buildings may distribute forces differently',
            'height_ft_pre_eq': 'Taller buildings experience stronger vibrations',
            'roof_type': 'Different roof materials respond differently to shaking',
            'foundation_type': 'Foundation quality is critical for earthquake resilience',
            'ground_floor_type': 'Ground floor design affects overall building stability',
            'position': 'Building position relative to terrain affects ground motion',
            'superstructure': 'Material and design of load-bearing elements critical',
            'plan_configuration': 'Building shape and configuration affect seismic response'
        }
        
        print(f"\nTop {n_top} Most Important Features:")
        print("=" * 70)
        for idx, (feature, importance) in enumerate(top_features.items(), 1):
            print(f"{idx}. {feature} (Importance: {importance:.4f})")
            print(f"   Interpretation: {interpretations.get(feature, 'Feature effect on damage')}\n")
    
    def plot_decision_tree_structure(self, max_depth=3):
        """
        Visualize decision tree structure for interpretability.
        """
        dt_model = self.model.named_steps['dt']
        
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(dt_model,
                 feature_names=self.feature_names,
                 class_names=['Not Severe', 'Severe'],
                 filled=True,
                 ax=ax,
                 max_depth=max_depth,
                 fontsize=10)
        
        plt.tight_layout()
        return fig


class HyperparameterExplanation:
    """
    Explain what each hyperparameter does and why we optimize them.
    """
    
    @staticmethod
    def explain_decision_tree_parameters():
        """
        Document what each Decision Tree parameter controls.
        """
        explanations = {
            'max_depth': {
                'what': 'Maximum depth of the tree',
                'why': 'Controls model complexity. Too deep = overfitting, too shallow = underfitting',
                'range': '5-20 for earthquake data',
                'effect_on_bias': 'Deeper trees: lower bias, higher variance'
            },
            'min_samples_split': {
                'what': 'Minimum samples required to split a node',
                'why': 'Prevents splits on very small subsets, avoiding noisy patterns',
                'range': '5-20 for earthquake data',
                'effect_on_bias': 'Higher values: simpler trees, higher bias, lower variance'
            },
            'min_samples_leaf': {
                'what': 'Minimum samples required in a leaf node',
                'why': 'Prevents creating pure leaves that overfit single examples',
                'range': '2-10 for earthquake data',
                'effect_on_bias': 'Higher values: smoother predictions, higher bias'
            },
            'criterion': {
                'what': 'Function to measure split quality',
                'why': 'Gini vs Entropy - different measures of impurity',
                'values': ['gini', 'entropy'],
                'effect': 'Usually similar results, Gini slightly faster'
            }
        }
        
        for param, details in explanations.items():
            print(f"\n{param.upper()}")
            print(f"  What: {details['what']}")
            print(f"  Why: {details['why']}")
            if 'range' in details:
                print(f"  Search Range: {details['range']}")
            if 'values' in details:
                print(f"  Values: {details['values']}")
            if 'effect_on_bias' in details:
                print(f"  Bias-Variance: {details['effect_on_bias']}")
    
    @staticmethod
    def explain_logistic_regression_parameters():
        """
        Document Logistic Regression hyperparameters.
        """
        explanations = {
            'C': {
                'what': 'Inverse of regularization strength',
                'interpretation': 'Lower C = stronger regularization (simpler model)',
                'higher_C': 'Lets model fit training data more closely',
                'lower_C': 'Encourages smaller coefficients, prevents overfitting',
                'range': '[0.001, 100]',
                'why': 'Prevents the model from becoming too complex'
            },
            'max_iter': {
                'what': 'Maximum iterations for solver to converge',
                'why': 'Logistic regression uses iterative optimization',
                'range': '[500, 1000]',
                'effect': 'Higher values = better convergence but longer training'
            }
        }
        
        for param, details in explanations.items():
            print(f"\n{param.upper()}")
            print(f"  What: {details['what']}")
            if 'interpretation' in details:
                print(f"  Interpretation: {details['interpretation']}")
            if 'why' in details:
                print(f"  Why: {details['why']}")
            if 'range' in details:
                print(f"  Search Range: {details['range']}")
