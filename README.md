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

4. Earthquake Damage Prediction in Nepal
   
   Built a classification model to predict building damage severity following the 2015 Nepal earthquake. Used data from the Kavrepalanchok district, focusing on distinguishing between severe and non-severe damage based on building structural characteristics.
   
   Key elements:
   - Queried SQLite database to join building structure and damage information
   - Handled categorical features with one-hot and ordinal encoding
   - Compared Logistic Regression and Decision Tree classifiers
   - Used validation curves to identify optimal tree depth
   - Extracted and visualized feature importance from the final model

5. Brazil and Mexico Real Estate Markets
   
   Analyzed real estate markets across different Latin American cities. In Mexico City specifically, applied feature engineering to extract geographic coordinates from lat-lon strings and borough information from hierarchical location data.
   
   Approach:
   - Handled missing values by filtering columns with >50% NaN values
   - Used ColumnTransformer to apply different preprocessing to categorical vs numerical features
   - Built pipelines with one-hot encoding for categorical variables and standard scaling for numerical ones
   - Compared model performance across different feature sets

6. Poland Bankruptcy Prediction
   
   End-to-end machine learning project predicting corporate bankruptcy. Dealt with a highly imbalanced dataset where bankruptcy cases were rare, requiring careful handling through oversampling.
   
   Workflow:
   - Loaded compressed JSON data and converted to pandas DataFrames
   - Identified class imbalance and applied random oversampling
   - Trained Random Forest classifier with cross-validation
   - Used GridSearchCV to optimize hyperparameters (max_depth, n_estimators)
   - Generated confusion matrix and classification reports
   - Extracted feature importance to identify the strongest bankruptcy predictors

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
- imblearn: Handling imbalanced datasets
- category_encoders: Advanced encoding strategies
- country_converter: Geographic data mapping


Competencies Demonstrated
--------------------------

Statistical Analysis: I've worked with hypothesis testing, power analysis, and chi-square tests. This includes determining required sample sizes for experiments and calculating statistical significance.

Machine Learning: Built both supervised (regression, classification) and unsupervised (clustering) models. Experience with model selection, hyperparameter tuning, and cross-validation strategies.

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
|-- bankruptcy/
|   |-- bankruptcy_poland.py
|-- market-analysis/
|   |-- market_volatility_india.py
|-- utils.py
|-- requirements.txt
|-- .gitignore
|-- README.md


Key Takeaways
-------------

This portfolio demonstrates the ability to tackle diverse data science challenges. The projects range from statistical experimentation to predictive modeling to unsupervised learning. They show not just knowledge of algorithms, but understanding of when and how to apply them to real problems.

I value clean, readable code with appropriate documentation. The ability to move from raw data to actionable insights through careful analysis and visualization is central to my work.


Contact
-------

GitHub: https://github.com/Harshit-sys169
Always interested in discussing data science, machine learning, or these projects.