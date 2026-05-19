# Utility functions and helpers for data science workflows
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


class DataProcessor:
    """Common data processing operations for real estate datasets."""
    
    @staticmethod
    def filter_real_estate_data(df, city=None, property_type='apartment', 
                               max_price=400000, min_area=20, max_area=500):
        """
        Apply standard filters to real estate datasets.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw real estate data
        city : str, optional
            Filter by city/region name
        property_type : str
            Type of property to filter (default: 'apartment')
        max_price : float
            Upper bound on price in USD
        min_area : float
            Lower bound on property area in m2
        max_area : float
            Upper bound on property area in m2
            
        Returns:
        --------
        pd.DataFrame
            Filtered dataset
        """
        filtered = df.copy()
        
        if 'property_type' in filtered.columns:
            filtered = filtered[filtered['property_type'] == property_type]
        
        if city and 'place_with_parent_names' in filtered.columns:
            filtered = filtered[filtered['place_with_parent_names'].str.contains(city, na=False)]
        
        if 'price_aprox_usd' in filtered.columns:
            filtered = filtered[filtered['price_aprox_usd'] <= max_price]
        
        if 'surface_covered_in_m2' in filtered.columns:
            filtered = filtered[
                (filtered['surface_covered_in_m2'] >= min_area) &
                (filtered['surface_covered_in_m2'] <= max_area)
            ]
        
        return filtered.dropna(subset=['price_aprox_usd', 'surface_covered_in_m2'])
    
    @staticmethod
    def remove_outliers_quantile(df, column, lower_q=0.1, upper_q=0.9):
        """
        Remove outliers using quantile bounds.
        Common approach for real estate data where extreme values skew analysis.
        """
        lower = df[column].quantile(lower_q)
        upper = df[column].quantile(upper_q)
        return df[df[column].between(lower, upper)]
    
    @staticmethod
    def extract_coordinates(df, latlon_col='lat-lon'):
        """
        Extract latitude and longitude from a combined column.
        """
        coords = df[latlon_col].str.split(',', expand=True)
        df['lat'] = pd.to_numeric(coords[0].str.strip(), errors='coerce')
        df['lon'] = pd.to_numeric(coords[1].str.strip(), errors='coerce')
        return df.drop(columns=[latlon_col])
    
    @staticmethod
    def extract_neighborhood(df, location_col='place_with_parent_names', sep='|', index=3):
        """
        Extract neighborhood from hierarchical location string.
        """
        df['neighborhood'] = df[location_col].str.split(sep, expand=True)[index]
        return df


class ModelEvaluator:
    """Helper functions for model evaluation and comparison."""
    
    @staticmethod
    def compare_models(y_true, predictions_dict, metric='mae'):
        """
        Compare multiple model predictions.
        
        Parameters:
        -----------
        y_true : array-like
            Ground truth values
        predictions_dict : dict
            Dictionary with model names as keys and predictions as values
        metric : str
            'mae' for mean absolute error (regression)
            'accuracy' for accuracy (classification)
        """
        from sklearn.metrics import mean_absolute_error, accuracy_score
        
        results = {}
        metric_func = mean_absolute_error if metric == 'mae' else accuracy_score
        
        for model_name, predictions in predictions_dict.items():
            results[model_name] = metric_func(y_true, predictions)
        
        return pd.Series(results).sort_values()
    
    @staticmethod
    def plot_residuals(y_true, y_pred, title='Residual Plot'):
        """
        Visualize model residuals to check for patterns.
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{title} - Predicted vs Residuals')
        axes[0].grid(True, alpha=0.3)
        
        # Distribution of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residual Value')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{title} - Residual Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


class FeatureAnalyzer:
    """Tools for feature engineering and analysis."""
    
    @staticmethod
    def get_feature_importance_summary(model, feature_names, top_n=10):
        """
        Extract and display top N important features from a trained model.
        Works with sklearn models that have feature_importances_ attribute.
        """
        if not hasattr(model, 'feature_importances_'):
            return None
        
        importance = pd.Series(
            model.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)
        
        return importance.head(top_n)
    
    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=15):
        """
        Visualize feature importance as a horizontal bar chart.
        """
        importance = FeatureAnalyzer.get_feature_importance_summary(
            model, feature_names, top_n=top_n
        )
        
        if importance is None:
            print("Model does not have feature importances.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        importance.sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def correlation_heatmap(df, numeric_only=True, figsize=(10, 8)):
        """
        Create correlation matrix heatmap for feature analysis.
        """
        if numeric_only:
            df = df.select_dtypes(include=[np.number])
        
        fig, ax = plt.subplots(figsize=figsize)
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        return fig


class ValidationHelper:
    """Cross-validation and validation curve utilities."""
    
    @staticmethod
    def plot_learning_curve(train_scores, val_scores, param_name, param_values):
        """
        Plot training vs validation performance across a parameter.
        Useful for identifying overfitting/underfitting.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(param_values, train_scores, marker='o', label='Training', linewidth=2)
        ax.plot(param_values, val_scores, marker='s', label='Validation', linewidth=2)
        ax.fill_between(param_values, train_scores, val_scores, alpha=0.2)
        
        ax.set_xlabel(param_name.replace('_', ' ').title())
        ax.set_ylabel('Score')
        ax.set_title(f'Learning Curve: {param_name.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig