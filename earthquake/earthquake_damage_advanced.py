import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    f1_score, accuracy_score, balanced_accuracy_score
)
import warnings
warnings.filterwarnings('ignore')


class NepalEarthquakePredictor:
    """
    Building damage prediction following 2015 Nepal earthquake.
    
    This is a real-world binary classification problem:
    - Target: Predicting severe building damage (Grade 3 or higher)
    - Challenge: Imbalanced classes in earthquake damage
    - Domain: Structural engineering + ML
    - Impact: Helps prioritize building reinforcement efforts
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.cv_results = None
        self.best_models = {}
        self.feature_names = None
    
    @staticmethod
    def load_and_wrangle(sqlite_path, district_id=3):
        """
        Load earthquake data from SQLite database.
        
        The Kavrepalanchok district (district_id=3) is used because:
        - Sufficient sample size for reliable model training
        - Good mix of building types and damage grades
        - Complete records with structural information
        """
        conn = sqlite3.connect(sqlite_path)
        
        query = f"""
        SELECT im.b_id,
               bs.age_building,
               bs.plinth_area_sq_ft,
               bs.height_ft_pre_eq,
               bs.land_surface_condition,
               bs.foundation_type,
               bs.roof_type,
               bs.ground_floor_type,
               bs.other_floor_type,
               bs.position,
               bs.plan_configuration,
               bs.superstructure,
               bd.damage_grade
        FROM (
            SELECT DISTINCT building_id AS b_id
            FROM id_map
            WHERE district_id = {district_id}
        ) im
        JOIN building_structure bs ON im.b_id = bs.building_id
        JOIN building_damage bd ON im.b_id = bd.building_id;
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Create binary target: severe damage (Grade 3+) vs not
        df['severe_damage'] = (df['damage_grade'] > 'Grade 3').astype(int)
        df = df.drop(columns=['damage_grade', 'b_id'])
        
        return df
    
    def analyze_data(self, df):
        """
        Exploratory data analysis and class distribution.
        """
        print("Dataset Information:")
        print(f"  Total buildings: {len(df)}")
        print(f"  Features: {df.shape[1] - 1}")
        
        class_dist = df['severe_damage'].value_counts()
        class_pct = df['severe_damage'].value_counts(normalize=True) * 100
        print(f"\nClass Distribution:")
        print(f"  Not Severe: {class_dist[0]} ({class_pct[0]:.1f}%)")
        print(f"  Severe: {class_dist[1]} ({class_pct[1]:.1f}%)")
        print(f"  Imbalance Ratio: {class_dist[0]/class_dist[1]:.1f}:1\n")
        
        return df
    
    def prepare_data(self, df, val_size=0.2, test_size=0.2):
        """
        Three-way split: train/val/test
        Using stratified split to preserve class distribution across all sets.
        """
        X = df.drop(columns='severe_damage')
        y = df['severe_damage']
        self.feature_names = X.columns.tolist()
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)  # Adjust ratio
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Data Split:")
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Validation set: {len(self.X_val)} samples")
        print(f"  Test set: {len(self.X_test)} samples\n")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def get_baseline_performance(self):
        """
        Calculate majority class baseline.
        Any model should beat this.
        """
        baseline_accuracy = max(self.y_train.value_counts()) / len(self.y_train)
        return baseline_accuracy
    
    def optimize_logistic_regression(self):
        """
        Grid search for optimal Logistic Regression parameters.
        
        Parameters explored:
        - C: Inverse regularization strength (higher C = less regularization)
        - max_iter: Maximum iterations for convergence
        """
        print("Optimizing Logistic Regression...")
        
        param_grid = {
            'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'lr__max_iter': [500, 1000]
        }
        
        pipeline = Pipeline([
            ('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ('lr', LogisticRegression(random_state=self.random_state))
        ])
        
        # Stratified K-Fold for balanced evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',  # F1 is better for imbalanced data than accuracy
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}\n")
        
        self.best_models['Logistic Regression'] = grid_search.best_estimator_
        return grid_search
    
    def optimize_decision_tree(self):
        """
        Grid search for optimal Decision Tree parameters.
        
        Parameters explored:
        - max_depth: Controls tree complexity (prevents overfitting)
        - min_samples_split: Minimum samples needed to split a node
        - min_samples_leaf: Minimum samples in leaf nodes
        - criterion: Gini vs entropy for split selection
        """
        print("Optimizing Decision Tree...")
        
        param_grid = {
            'dt__max_depth': range(5, 20, 2),
            'dt__min_samples_split': [5, 10, 20],
            'dt__min_samples_leaf': [2, 5, 10],
            'dt__criterion': ['gini', 'entropy']
        }
        
        pipeline = Pipeline([
            ('encoding', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ('dt', DecisionTreeClassifier(random_state=self.random_state))
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 Score: {grid_search.best_score_:.4f}\n")
        
        self.best_models['Decision Tree'] = grid_search.best_estimator_
        return grid_search
    
    def cross_validate_models(self):
        """
        Perform cross-validation with multiple metrics.
        
        Why cross-validation?
        - Single train/test split gives lucky/unlucky results
        - CV uses multiple folds to get robust estimate
        - Stratified K-Fold preserves class distribution
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'accuracy': 'accuracy',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        results = {}
        
        for model_name, model in self.best_models.items():
            cv_results = cross_validate(
                model,
                self.X_train,
                self.y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True
            )
            results[model_name] = cv_results
        
        self.cv_results = results
        return results
    
    def plot_cv_results(self):
        """
        Visualize cross-validation scores across folds.
        Shows both training and validation to detect overfitting.
        """
        metrics = ['accuracy', 'f1', 'roc_auc']
        fig, axes = plt.subplots(len(self.cv_results), len(metrics), figsize=(15, 10))
        
        for model_idx, (model_name, cv_result) in enumerate(self.cv_results.items()):
            for metric_idx, metric in enumerate(metrics):
                ax = axes[model_idx, metric_idx] if len(self.cv_results) > 1 else axes[metric_idx]
                
                train_scores = cv_result[f'train_{metric}']
                test_scores = cv_result[f'test_{metric}']
                folds = range(1, len(train_scores) + 1)
                
                ax.plot(folds, train_scores, 'o-', label='Train', linewidth=2)
                ax.plot(folds, test_scores, 's-', label='Validation', linewidth=2)
                ax.fill_between(folds, train_scores, test_scores, alpha=0.1)
                
                ax.set_xlabel('Fold')
                ax.set_ylabel('Score')
                ax.set_title(f'{model_name} - {metric.replace("_", " ").title()}')
                ax.set_ylim([0, 1])
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def evaluate_on_test_set(self):
        """
        Final evaluation on held-out test set.
        """
        results = {}
        
        for model_name, model in self.best_models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            results[model_name] = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'F1 Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
                'Precision': classification_report(self.y_test, y_pred, output_dict=True)['1']['precision'],
                'Recall': classification_report(self.y_test, y_pred, output_dict=True)['1']['recall']
            }
        
        return pd.DataFrame(results).T
    
    def plot_confusion_matrices(self):
        """
        Side-by-side confusion matrices for comparison.
        """
        fig, axes = plt.subplots(1, len(self.best_models), figsize=(12, 4))
        
        for idx, (model_name, model) in enumerate(self.best_models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            ax = axes[idx] if len(self.best_models) > 1 else axes
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Severe', 'Severe'],
                       yticklabels=['Not Severe', 'Severe'])
            ax.set_title(f'{model_name}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    print("Nepal Earthquake Damage Prediction - Advanced Analysis")
    print("=" * 70)
    print()
    
    # Load and explore data
    df = NepalEarthquakePredictor.load_and_wrangle('../nepal.sqlite')
    
    predictor = NepalEarthquakePredictor()
    predictor.analyze_data(df)
    
    # Prepare data with stratification
    predictor.prepare_data(df)
    baseline = predictor.get_baseline_performance()
    print(f"Baseline Accuracy (majority class): {baseline:.4f}\n")
    
    # Hyperparameter optimization
    print("Hyperparameter Optimization Phase")
    print("-" * 70)
    predictor.optimize_logistic_regression()
    predictor.optimize_decision_tree()
    
    # Cross-validation
    print("\nCross-Validation Phase")
    print("-" * 70)
    predictor.cross_validate_models()
    
    # Test set evaluation
    print("\nFinal Test Set Performance")
    print("-" * 70)
    test_results = predictor.evaluate_on_test_set()
    print(test_results)
