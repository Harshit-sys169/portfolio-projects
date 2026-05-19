import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


class BusinessMetricsAnalyzer:
    """
    Connect model predictions to business impact.
    
    A model's statistical metrics (accuracy, precision, recall) matter
    only insofar as they translate to business value.
    """
    
    def __init__(self, cost_fn=100000, cost_fp=5000):
        """
        Initialize with business costs.
        
        Parameters:
        -----------
        cost_fn : float
            Cost of false negative (missing actual bankruptcy) in dollars
            Higher because undetected bankruptcies cause financial loss
        cost_fp : float
            Cost of false positive (false alarm) in dollars
            Lower because it triggers investigation, which is cheaper than bankruptcy
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
    
    def calculate_business_impact(self, y_true, y_pred):
        """
        Calculate total cost of predictions based on confusion matrix.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = (fn * self.cost_fn) + (fp * self.cost_fp)
        
        return {
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp,
            'FN Cost': fn * self.cost_fn,
            'FP Cost': fp * self.cost_fp,
            'Total Cost': total_cost,
            'Cost per False Negative': self.cost_fn,
            'Cost per False Positive': self.cost_fp
        }
    
    def optimize_for_business_metrics(self, models, y_pred_proba_dict, y_true):
        """
        Compare models not just on statistical metrics, but on business costs.
        """
        results = {}
        
        for model_name, y_pred_proba in y_pred_proba_dict.items():
            # Try different thresholds
            costs = []
            thresholds = np.arange(0.1, 0.9, 0.05)
            
            for thresh in thresholds:
                y_pred = (y_pred_proba >= thresh).astype(int)
                impact = self.calculate_business_impact(y_true, y_pred)
                costs.append(impact['Total Cost'])
            
            min_cost_idx = np.argmin(costs)
            optimal_threshold = thresholds[min_cost_idx]
            min_cost = costs[min_cost_idx]
            
            results[model_name] = {
                'optimal_threshold': optimal_threshold,
                'minimum_cost': min_cost,
                'cost_per_detection': min_cost / max(np.sum(y_true), 1)
            }
        
        return pd.DataFrame(results).T
    
    def plot_cost_curves(self, y_pred_proba, y_true):
        """
        Visualize how total cost changes with decision threshold.
        """
        thresholds = np.arange(0.1, 0.95, 0.01)
        fn_costs = []
        fp_costs = []
        total_costs = []
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            impact = self.calculate_business_impact(y_true, y_pred)
            fn_costs.append(impact['FN Cost'])
            fp_costs.append(impact['FP Cost'])
            total_costs.append(impact['Total Cost'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, fn_costs, label='False Negative Cost', linewidth=2)
        ax.plot(thresholds, fp_costs, label='False Positive Cost', linewidth=2)
        ax.plot(thresholds, total_costs, label='Total Cost', linewidth=2.5, color='black')
        
        min_cost_idx = np.argmin(total_costs)
        ax.axvline(thresholds[min_cost_idx], color='red', linestyle='--',
                   label=f'Optimal Threshold: {thresholds[min_cost_idx]:.2f}')
        
        ax.set_xlabel('Decision Threshold')
        ax.set_ylabel('Cost ($)')
        ax.set_title('Business Cost vs. Decision Threshold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        return fig
