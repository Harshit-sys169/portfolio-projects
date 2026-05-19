import gzip
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, balanced_accuracy_score
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")


class BankruptcyPredictor:
    """
    Comprehensive bankruptcy prediction system with advanced ensemble methods
    and interpretability analysis.
    
    This implementation addresses real-world challenges:
    - Highly imbalanced datasets (most companies don't go bankrupt)
    - Need for both predictive accuracy and business interpretability
    - Cost asymmetry (false negatives are more costly than false positives)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
    
    @staticmethod
    def load_data(filepath):
        """
        Load compressed JSON bankruptcy data.
        """
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data['observations'])
        df = df.set_index('id')
        return df
    
    def prepare_data(self, df, test_size=0.2):
        """
        Split data and analyze class balance.
        """
        X = df.drop(columns='bankrupt')
        y = df['bankrupt']
        
        # Display class distribution
        class_dist = y.value_counts()
        class_pct = y.value_counts(normalize=True) * 100
        print("Class Distribution:")
        print(f"  Non-Bankrupt: {class_dist[0]} ({class_pct[0]:.2f}%)")
        print(f"  Bankrupt: {class_dist[1]} ({class_pct[1]:.2f}%)")
        print(f"  Imbalance Ratio: {class_dist[0]/class_dist[1]:.1f}:1\n")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_imbalance_oversampling(self):
        """
        Use Random Oversampling to balance training data.
        
        Pros: Simple, preserves original data
        Cons: Risk of overfitting on minority class
        """
        oversampler = RandomOverSampler(random_state=self.random_state)
        X_resampled, y_resampled = oversampler.fit_resample(self.X_train, self.y_train)
        return X_resampled, y_resampled
    
    def handle_imbalance_smote(self):
        """
        Use SMOTE (Synthetic Minority Oversampling Technique).
        
        Generates synthetic examples in the minority class by interpolating
        between existing minority samples. More sophisticated than random oversampling.
        """
        smote = SMOTE(random_state=self.random_state, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        return X_resampled, y_resampled
    
    def build_baseline_model(self, X_train, y_train):
        """
        Logistic Regression with class weighting for imbalanced data.
        
        Cost-sensitive learning: assigns higher penalties to misclassifying
        the minority (bankrupt) class.
        """
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Automatically adjusts weights
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        return model
    
    def build_ensemble_models(self, X_train, y_train):
        """
        Build multiple ensemble methods for comparison.
        Each has different strengths for imbalanced data.
        """
        models_dict = {}
        
        # Random Forest with class weighting
        models_dict['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Gradient Boosting - focuses on hard examples
        models_dict['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=self.random_state
        )
        
        # AdaBoost - specifically designed for imbalanced problems
        models_dict['AdaBoost'] = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        # Train all models
        for name, model in models_dict.items():
            model.fit(X_train, y_train)
            self.models[name] = model
        
        return models_dict
    
    def evaluate_models(self):
        """
        Comprehensive model evaluation using multiple metrics.
        
        For imbalanced data, we focus on:
        - ROC-AUC: Robust to class imbalance
        - F1 Score: Harmonic mean of precision and recall
        - Balanced Accuracy: Average recall for each class
        - Precision/Recall: Business-relevant tradeoff
        """
        results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            results[model_name] = {
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
                'F1 Score': f1_score(self.y_test, y_pred),
                'Balanced Accuracy': balanced_accuracy_score(self.y_test, y_pred),
                'Precision': classification_report(self.y_test, y_pred, output_dict=True)['1']['precision'],
                'Recall': classification_report(self.y_test, y_pred, output_dict=True)['1']['recall']
            }
        
        self.results = pd.DataFrame(results).T
        return self.results
    
    def plot_model_comparison(self):
        """
        Visualize model performance across metrics.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = self.results.columns
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            self.results[metric].sort_values(ascending=False).plot(
                kind='barh', ax=ax, color='steelblue'
            )
            ax.set_title(f'{metric}')
            ax.set_xlabel('Score')
        
        # Remove last empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self):
        """
        Compare ROC curves across all models.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc = roc_auc_score(self.y_test, y_pred_proba)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve Comparison')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curves(self):
        """
        Precision-Recall curves are more informative for imbalanced data
        than ROC curves because they focus on the minority class.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            ax.plot(recall, precision, label=model_name, linewidth=2)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve Comparison')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        return fig
    
    def explain_with_shap(self, model_name='Gradient Boosting'):
        """
        Use SHAP (SHapley Additive exPlanations) for model interpretability.
        
        SHAP values provide consistent, theoretically sound feature importance
        that answers: 'How much did each feature contribute to this prediction?'
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        model = self.models[model_name]
        
        # Create SHAP explainer
        # For tree-based models, use TreeExplainer (fast and exact)
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, self.X_test.iloc[:100])
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(self.X_test)
        
        # For binary classification, take positive class shap values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return explainer, shap_values
    
    def plot_shap_summary(self, explainer, shap_values):
        """
        Plot SHAP summary showing feature importance and direction.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, plot_type='bar', show=False)
        return fig
    
    def get_feature_importance(self, model_name='Gradient Boosting'):
        """
        Extract feature importance from tree-based models.
        """
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.Series(
                model.feature_importances_,
                index=self.X_train.columns
            ).sort_values(ascending=False)
            return importance
        return None
    
    def plot_confusion_matrix(self, model_name='Gradient Boosting'):
        """
        Visualize confusion matrix for the best model.
        For imbalanced data, understand both false positives and false negatives.
        """
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Non-Bankrupt', 'Bankrupt'],
                    yticklabels=['Non-Bankrupt', 'Bankrupt'])
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig


if __name__ == "__main__":
    print("Poland Bankruptcy Prediction - Advanced Analysis")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading data...")
    df = BankruptcyPredictor.load_data('data/taiwan-bankruptcy-data.json.gz')
    print(f"Dataset shape: {df.shape}\n")
    
    # Initialize predictor
    predictor = BankruptcyPredictor()
    predictor.prepare_data(df)
    
    # Handle imbalance with SMOTE
    print("Applying SMOTE for imbalanced data handling...")
    X_train_balanced, y_train_balanced = predictor.handle_imbalance_smote()
    print(f"Balanced training set shape: {X_train_balanced.shape}\n")
    
    # Train baseline model
    print("Training baseline model (Logistic Regression with class weighting)...")
    baseline = predictor.build_baseline_model(predictor.X_train, predictor.y_train)
    predictor.models['Logistic Regression'] = baseline
    
    # Train ensemble models
    print("Training ensemble methods...")
    predictor.build_ensemble_models(X_train_balanced, y_train_balanced)
    print(f"Models trained: {list(predictor.models.keys())}\n")
    
    # Evaluate
    print("Evaluating models...")
    results = predictor.evaluate_models()
    print(results)
    print()
    
    # Feature importance
    print("Top 10 Most Important Features (Gradient Boosting):")
    importance = predictor.get_feature_importance('Gradient Boosting')
    if importance is not None:
        print(importance.head(10))
        print()
    
    # SHAP explanation
    if SHAP_AVAILABLE:
        print("Generating SHAP explanations...")
        explainer, shap_vals = predictor.explain_with_shap('Gradient Boosting')
        print("SHAP values computed successfully")
    else:
        print("SHAP analysis skipped (library not installed)")
