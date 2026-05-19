Portfolio Projects
==================

A collection of data science and quantitative finance projects demonstrating practical applications of machine learning, statistical analysis, and real estate modeling.


Projects
--------

1. A/B Testing at World Quant University
   
   Worked on statistical hypothesis testing and experimental design within the context of financial applications. The project involved designing an experiment to test the effectiveness of different applicant engagement strategies. I calculated statistical power requirements to determine sample sizes, implemented chi-square tests for statistical significance, and analyzed contingency tables to understand the relationship between treatment groups and outcomes.
   
   Key aspects:
   - Power analysis using chi-square distribution to determine required sample sizes
   - MongoDB database integration for experiment data management
   - Contingency table analysis and odds ratio calculations
   - Visualization of experimental results using geographic choropleths
   - Date-based filtering and aggregation of applicant data

2. Buenos Aires Real Estate Analysis (4 Tasks)
   
   A comprehensive real estate analysis broken into phases of increasing complexity. Started with basic linear regression on apartment prices based on surface area, then expanded to include geographic features (latitude/longitude), neighborhood data, and ultimately built a multi-feature pipeline comparing different regression approaches.
   
   Task progression:
   - Task 1: Simple linear regression on apartment features
   - Task 2: Geographic-based pricing using lat/lon coordinates with 3D visualization
   - Task 3: Neighborhood analysis with one-hot encoding and Ridge regression
   - Task 4: Full pipeline comparison of Linear vs Ridge regression across all features
   
   Data involved filtering for Capital Federal region apartments under $400k USD, removing outliers using quantile-based bounds, and handling missing values appropriately.

3. Credit Risk Analysis in the US
   
   Unsupervised learning project focused on clustering small business owners based on financial characteristics. Applied K-Means clustering with standardization, then used Principal Component Analysis (PCA) to visualize high-dimensional credit risk data in 2D space.
   
   Process:
   - Identified high-variance features related to business owner finances
   - Experimented with different numbers of clusters (2-12) and tracked both inertia and silhouette scores
   - Selected optimal cluster count based on silhouette analysis
   - Performed inverse transformation to interpret cluster characteristics in original feature space
   - Visualized both the data distribution and PCA decomposition

4. Earthquake Damage Prediction in Nepal (Advanced)
   
   Built a comprehensive classification model to predict building damage severity following the 2015 Nepal earthquake, with focus on rigorous model development methodology.
   
   Core Analysis (earthquake_damage_advanced.py):
   - Queried SQLite database to join building structure and damage information
   - Analyzed class distribution and imbalance characteristics
   - Three-way stratified split (train/val/test) to preserve class balance
   - Compared Logistic Regression and Decision Tree classifiers
   
   Feature Importance Analysis (feature_importance_analysis.py):
   - Tree-based importance: How features contribute to split decisions
   - Permutation importance: How predictions degrade when features are shuffled
   - Feature group analysis: Understanding importance of structural categories
   - Business interpretation: What each top feature means for earthquake resilience
   - Decision tree visualization: Seeing the actual decision logic
   
   Hyperparameter Optimization (earthquake_damage_advanced.py):
   For Decision Trees:
   - max_depth (5-20): Controls tree complexity and prevents overfitting
   - min_samples_split (5-20): Prevents splits on very small groups
   - min_samples_leaf (2-10): Prevents pure leaves that memorize single examples
   - criterion (gini vs entropy): Different measures of split quality
   
   For Logistic Regression:
   - C (0.001-100): Regularization strength - lower C = simpler model
   - max_iter (500-1000): Convergence iterations for optimization
   
   Grid search explores all combinations using stratified K-Fold cross-validation
   and F1 score as the optimization metric (better than accuracy for imbalanced data).
   
   Cross-Validation Strategy (cross_validation_explanation.py):
   - Why: Single train/test split is unreliable - results vary with random seed
   - What: Stratified K-Fold (5 folds) ensures each fold has same class distribution
   - Metrics used:
     * F1 Score: Balances precision and recall, primary metric for imbalanced data
     * ROC-AUC: Performance across all decision thresholds
     * Balanced Accuracy: Average recall for each class
     * Precision/Recall: Tradeoff visualization
   - Interpretation: High variance across folds indicates model is sensitive to data
   
   Result: Rigorous, reproducible methodology suitable for production deployment.

5. Brazil and Mexico Real Estate Markets
   
   Analyzed real estate markets across different Latin American cities. In Mexico City specifically, applied feature engineering to extract geographic coordinates from lat-lon strings and borough information from hierarchical location data.
   
   Approach:
   - Handled missing values by filtering columns with >50% NaN values
   - Used ColumnTransformer to apply different preprocessing to categorical vs numerical features
   - Built pipelines with one-hot encoding for categorical variables and standard scaling for numerical ones
   - Compared model performance across different feature sets

6. Poland Bankruptcy Prediction (Advanced Analysis)
   
   End-to-end machine learning project predicting corporate bankruptcy with advanced techniques for handling real-world challenges.
   
   This analysis goes beyond basic classification to address practical concerns:
   
   Imbalanced Data Handling:
   - Explored multiple techniques: Random Oversampling, SMOTE (Synthetic Minority Oversampling Technique)
   - SMOTE creates synthetic examples by interpolating between minority samples, reducing overfitting risk
   - Demonstrated how imbalance ratios affect model training and evaluation
   
   Ensemble Methods Comparison:
   - Random Forest with class weighting for balanced learning
   - Gradient Boosting that naturally focuses on hard-to-classify examples
   - AdaBoost designed specifically for handling difficult cases
   - Baseline Logistic Regression with cost-sensitive class weighting
   
   Model Evaluation for Imbalanced Data:
   - ROC-AUC: Metric robust to class imbalance
   - F1 Score: Harmonic mean of precision and recall
   - Balanced Accuracy: Average recall for each class, not skewed by majority class
   - Precision-Recall curves more informative than ROC for imbalanced problems
   
   Cost-Sensitive Learning:
   - Different misclassification errors have different business costs
   - False Negatives (missing bankruptcies) are more costly than False Positives
   - Optimized decision thresholds based on business metrics, not statistical accuracy
   - Analyzed how prediction costs vary with classification threshold
   
   Model Interpretability:
   - Feature importance from tree-based models
   - SHAP (SHapley Additive exPlanations) for detailed prediction explanations
   - Confusion matrices to understand error patterns
   - Business impact analysis connecting predictions to financial consequences
   
   Files included:
   - bankruptcy_poland_advanced.py: Core analysis with all ensemble methods and SHAP
   - cost_sensitive_learning.py: Handling asymmetric misclassification costs
   - business_metrics.py: Translating model predictions to business value

7. Market Volatility in India
   
   Financial market analysis module for studying volatility patterns and market trends. Includes functions for loading market data, calculating rolling volatility, and visualizing trends over time.


Technology Stack
----------------

Python is the primary language across all projects.

Core Libraries:
- pandas: Data manipulation and analysis
- numpy: Numerical operations and array handling
- scikit-learn: Machine learning models and preprocessing
- scipy & statsmodels: Statistical analysis and hypothesis testing

Data & Database:
- SQLite: Querying relational databases
- MongoDB: Document-based data storage and aggregation
- SQLAlchemy: Database abstraction layer

Visualization:
- matplotlib: Static plotting
- seaborn: Statistical visualizations
- plotly: Interactive visualizations

Specialized:
- imblearn: Handling imbalanced datasets (SMOTE, oversampling)
- shap: Model interpretability and feature importance
- category_encoders: Advanced encoding strategies
- country_converter: Geographic data mapping


Competencies Demonstrated
--------------------------

Rigorous Model Development: The earthquake project demonstrates proper ML methodology including stratified splitting, hyperparameter optimization with cross-validation, and comprehensive evaluation. Not just fitting a model, but building one that generalizes reliably.

Feature Engineering and Analysis: Understanding which features matter and why. The earthquake project analyzes importance through multiple lenses (tree-based, permutation, business interpretation).

Statistical Analysis: Hypothesis testing, power analysis, chi-square tests. Understanding required sample sizes for experiments and calculating statistical significance.

Handling Real-World Challenges: Not just textbook problems. The bankruptcy project specifically demonstrates handling imbalanced data, cost asymmetry in misclassification, and translating statistical models to business value.

Ensemble Methods: Compared multiple approaches (Random Forest, Gradient Boosting, AdaBoost) and understood their different strengths for different problem characteristics.

Model Interpretability: Beyond just "this model gets 95% accuracy". Using SHAP values to understand what features drive predictions and how to explain them to stakeholders.

Hyperparameter Optimization: Grid search with proper cross-validation. Not just accepting defaults but understanding what each parameter controls and why it matters.

Data Engineering: Proficient at data cleaning, handling missing values, feature engineering, and building reusable pipelines. Comfortable working with structured data from databases and unstructured JSON data.

Problem Solving: Each project targets a real business question. Rather than toy datasets, these involve messy real-world data with appropriate handling of missing values, outliers, and class imbalance.

Visualization: Created meaningful visualizations that tell a story about the data. This includes everything from basic histograms and scatter plots to 3D surface plots and geographic choropleths.


Getting Started
---------------

To explore these projects:

1. Clone the repository:
   git clone https://github.com/Harshit-sys169/portfolio-projects.git
   cd portfolio-projects

2. Install dependencies:
   pip install -r requirements.txt

3. Each project directory is self-contained. Navigate to a project and examine the Python files.

For advanced bankruptcy analysis, install additional dependencies:
   pip install shap


Project Structure
-----------------

portfolio-projects/
|-- ab-testing/
|   |-- a_by_b_testing_wqu.py
|-- real-estate/
|   |-- buenos-aires/
|   |   |-- task_1.py
|   |   |-- task_2.py
|   |   |-- task_3.py
|   |   |-- task_4.py
|   |-- brazil-mexico/
|       |-- mexico.py
|       |-- mexico_2.py
|-- credit-risk/
|   |-- credit_risk_us.py
|-- earthquake/
|   |-- earthquake_damage_nepal.py
|   |-- earthquake_damage_advanced.py
|   |-- feature_importance_analysis.py
|   |-- cross_validation_explanation.py
|-- bankruptcy/
|   |-- bankruptcy_poland_advanced.py
|   |-- cost_sensitive_learning.py
|   |-- business_metrics.py
|-- market-analysis/
|   |-- market_volatility_india.py
|-- utils.py
|-- requirements.txt
|-- .gitignore
|-- README.md


Key Takeaways
-------------

This portfolio demonstrates the ability to tackle diverse data science challenges using rigorous methodology. The projects range from statistical experimentation to predictive modeling to unsupervised learning.

A key differentiator is the emphasis on proper ML practices: rigorous cross-validation, hyperparameter optimization, feature importance analysis, and translating technical results to business value. This shows not just knowledge of algorithms, but understanding of production-grade data science.

Recent enhancements show growing sophistication: the earthquake project demonstrates how to build a truly reproducible, generalizable model, while the bankruptcy project shows understanding that accuracy metrics matter less than business outcomes.


Contact
-------

GitHub: https://github.com/Harshit-sys169
Always interested in discussing data science, machine learning, or these projects.
