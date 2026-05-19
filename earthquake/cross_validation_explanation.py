import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score


class CrossValidationStrategy:
    """
    Comprehensive explanation and implementation of cross-validation for imbalanced data.
    
    Why Cross-Validation?
    =====================
    A single train/test split can be misleading:
    - One split might get a lucky/unlucky partitioning
    - Model performance estimate has high variance
    - Different splits can show very different results
    
    Cross-validation reduces this variance by:
    - Testing on multiple non-overlapping test sets
    - Using the same data more efficiently
    - Providing multiple performance estimates
    """
    
    @staticmethod
    def explain_stratified_kfold():
        """
        Explain why Stratified K-Fold is better than standard K-Fold for imbalanced data.
        """
        print("Cross-Validation Strategy: Stratified K-Fold")
        print("=" * 70)
        print()
        print("Standard K-Fold:")
        print("  - Randomly splits data into k equal-sized folds")
        print("  - Problem: Each fold might have different class distributions")
        print("  - Risk: Validation scores could vary wildly due to fold imbalance")
        print()
        print("Stratified K-Fold:")
        print("  - Ensures each fold has similar class distribution as full dataset")
        print("  - Splits proportionally: if dataset is 75% non-severe, each fold is ~75%")
        print("  - Result: More reliable and stable performance estimates")
        print()
        print("For Earthquake Data:")
        print("  - Severe damage buildings: ~27% of dataset")
        print("  - Each fold will have ~27% severe buildings")
        print("  - Prevents one fold getting all easy cases, another all hard cases")
        print()
    
    @staticmethod
    def demonstrate_fold_distribution(y, n_splits=5):
        """
        Show how Stratified K-Fold maintains class distribution.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        print(f"\nStratified {n_splits}-Fold Distribution:")
        print("=" * 70)
        print(f"Original distribution:")
        print(f"  Not Severe: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
        print(f"  Severe: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
        print()
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            print(f"Fold {fold_idx + 1}:")
            print(f"  Training: Not Severe {(y_train_fold==0).sum()} ({(y_train_fold==0).mean()*100:.1f}%), "
                  f"Severe {(y_train_fold==1).sum()} ({(y_train_fold==1).mean()*100:.1f}%)")
            print(f"  Validation: Not Severe {(y_val_fold==0).sum()} ({(y_val_fold==0).mean()*100:.1f}%), "
                  f"Severe {(y_val_fold==1).sum()} ({(y_val_fold==1).mean()*100:.1f}%)")
        print()
    
    @staticmethod
    def explain_cv_metrics():
        """
        Explain which metrics matter for cross-validation on imbalanced data.
        """
        print("\nMetrics for Cross-Validation (Imbalanced Data):")
        print("=" * 70)
        
        metrics_info = {
            'Accuracy': {
                'formula': '(TP + TN) / Total',
                'good_for': 'Balanced datasets',
                'problem': 'Useless for imbalanced data - always high when predicting majority class',
                'example': 'Model predicts "no damage" always: 73% accuracy but useless'
            },
            'F1 Score': {
                'formula': '2 * (Precision * Recall) / (Precision + Recall)',
                'good_for': 'Imbalanced data - balances precision and recall',
                'advantage': 'Penalizes both false positives and false negatives',
                'use': 'Primary metric for this earthquake problem'
            },
            'ROC-AUC': {
                'formula': 'Area under curve of TPR vs FPR',
                'good_for': 'Imbalanced data - shows performance across all thresholds',
                'advantage': 'Robust to class imbalance, independent of threshold',
                'use': 'Secondary metric to understand tradeoff between TP and FP'
            },
            'Balanced Accuracy': {
                'formula': '(Sensitivity + Specificity) / 2',
                'good_for': 'Imbalanced data - average recall for each class',
                'advantage': 'Each class weighted equally regardless of frequency',
                'use': 'Alternative to F1, shows per-class performance'
            }
        }
        
        for metric, info in metrics_info.items():
            print(f"\n{metric}:")
            print(f"  Formula: {info['formula']}")
            print(f"  Good for: {info['good_for']}")
            if 'problem' in info:
                print(f"  Problem: {info['problem']}")
            if 'advantage' in info:
                print(f"  Advantage: {info['advantage']}")
            if 'example' in info:
                print(f"  Example: {info['example']}")
            if 'use' in info:
                print(f"  Usage: {info['use']}")
    
    @staticmethod
    def plot_cv_scores(cv_scores_dict, metric_name='F1 Score'):
        """
        Visualize cross-validation scores across folds for multiple models.
        """
        fig, axes = plt.subplots(1, len(cv_scores_dict), figsize=(15, 5))
        
        if len(cv_scores_dict) == 1:
            axes = [axes]
        
        for idx, (model_name, scores) in enumerate(cv_scores_dict.items()):
            ax = axes[idx]
            folds = range(1, len(scores) + 1)
            
            ax.plot(folds, scores, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax.axhline(np.mean(scores), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(scores):.3f}')
            ax.fill_between(folds, np.mean(scores) - np.std(scores),
                           np.mean(scores) + np.std(scores), alpha=0.2)
            
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{model_name}\n(Std: {np.std(scores):.3f})')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def interpret_cv_results(cv_scores_dict):
        """
        Provide interpretation of cross-validation results.
        """
        print("\nCross-Validation Results Interpretation:")
        print("=" * 70)
        
        for model_name, scores in cv_scores_dict.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv_coeff = std_score / mean_score if mean_score > 0 else np.inf
            
            print(f"\n{model_name}:")
            print(f"  Mean Score: {mean_score:.4f}")
            print(f"  Std Dev: {std_score:.4f}")
            print(f"  Score Range: [{np.min(scores):.4f}, {np.max(scores):.4f}]")
            
            if cv_coeff < 0.05:
                stability = "Very Stable"
            elif cv_coeff < 0.10:
                stability = "Stable"
            elif cv_coeff < 0.15:
                stability = "Moderate Variability"
            else:
                stability = "High Variability"
            
            print(f"  Stability: {stability}")
            
            if std_score > mean_score * 0.1:
                print(f"  Warning: High variance across folds - model is sensitive to data split")
            else:
                print(f"  Good: Consistent performance across different data splits")
