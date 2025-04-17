# Model Performance Results

## Model Performance Comparison

| Model | RMSE | Best Parameters |
|-------|------|-----------------|
| CatBoost | 1025.52 | {'depth': 6, 'iterations': 300, 'l2_leaf_reg': 3, 'learning_rate': 0.05} |
| LightGBM | 1055.26 | {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300, 'num_leaves': 31} |
| Gradient Boosting | 1083.26 | {'learning_rate': 0.05, 'max_depth': 6, 'n_estimators': 300} |
| Weighted Voting Regressor | 1043.79 | {'weights': {'CatBoost': 0.3426387259205692, 'LightGBM': 0.3329840089964553, 'Gradient Boosting': 0.3243772650829755}} |

## Best Model Details

- Best Model: CatBoost
- RMSE: 1025.52
- Parameters: {'depth': 6, 'iterations': 300, 'l2_leaf_reg': 3, 'learning_rate': 0.05}

## Feature Importance Analysis

Top 10 Most Important Features:

| Feature | Importance |
|---------|------------|
| Item_MRP_Outlet_Type | 23.6751 |
| Outlet_Type_Mean_Sales | 17.8069 |
| Item_MRP | 14.2029 |
| Outlet_Type | 8.7414 |
| Item_MRP_Clusters | 5.2108 |
| Price_Range_Mean_Sales | 3.6821 |
| Outlet_Age_Group_Outlet_Type | 3.5064 |
| Item_Visibility | 2.9789 |
| Item_Outlet_Interaction | 2.4261 |
| Item_Weight | 2.3419 |

## Data Preprocessing Summary

1. Missing Value Handling:
   - Item_Weight: Filled with mean
   - Outlet_Size: Filled with 'Medium'

2. Feature Engineering:
   - Created Item_Type_Combined from Item_Identifier
   - Created Item_MRP_Clusters using quantiles
   - Created Item_Category based on Item_Type
   - Created Item_Type_New for perishable/non-perishable items
   - Created Outlet_Age and Outlet_Age_Group
   - Created Price_Range using quantiles
   - Created various interaction features

3. Feature Encoding:
   - Label encoded all categorical variables
   - Applied target encoding for categorical variables

4. Feature Scaling:
   - Scaled numerical features using StandardScaler

## Model Tuning Summary

1. Base Models:
   - CatBoost: RMSE = 1025.52
   - LightGBM: RMSE = 1055.26
   - Gradient Boosting: RMSE = 1083.26

2. Ensemble Models:
   - Weighted Voting Regressor: RMSE = 1043.79
