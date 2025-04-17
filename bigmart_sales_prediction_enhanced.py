import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import VotingRegressor, StackingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from tqdm import tqdm
import time
import os

# Add progress tracking
def print_progress(step, total_steps, message, start_time=None):
    elapsed = time.time() - start_time if start_time else 0
    print(f"\nStep {step}/{total_steps}: {message}")
    if start_time:
        print(f"Time elapsed: {elapsed:.2f} seconds")
    print("-" * 50)

# Load data
print_progress(1, 7, "Loading data...", time.time())
train = pd.read_csv('train_v9rqX0R.csv')
test = pd.read_csv('test_AbJTz2l.csv')
submission = pd.read_csv('sample_submission_8RXa3c6.csv')
print("✓ Data loaded successfully!")

# Store target encoding mappings and feature columns
target_encoding_maps = {}
train_feature_columns = None

def preprocess_data(df, is_train=True):
    global train_feature_columns
    
    # Handle missing values
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    df['Outlet_Size'] = df['Outlet_Size'].fillna('Medium')
    
    # Create features
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[:2])
    df['Item_MRP_Clusters'] = pd.qcut(df['Item_MRP'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    df['Item_Category'] = df['Item_Type'].apply(lambda x: 'Food' if x in ['Baking Goods', 'Breads', 'Breakfast', 'Dairy', 'Fruits and Vegetables', 'Meat', 'Seafood', 'Snack Foods', 'Starchy Foods'] else 'Non-Food')
    df['Item_Type_New'] = df['Item_Type'].apply(lambda x: 'Perishable' if x in ['Breads', 'Breakfast', 'Dairy', 'Fruits and Vegetables', 'Meat', 'Seafood'] else 'Non-Perishable')
    df['Outlet_Age'] = 2023 - df['Outlet_Establishment_Year']
    df['Outlet_Age_Group'] = pd.cut(df['Outlet_Age'], bins=[0, 10, 20, 30], labels=['New', 'Medium', 'Old'])
    df['Price_Range'] = pd.qcut(df['Item_MRP'], q=3, labels=['Low', 'Medium', 'High'])
    
    # Encode categorical variables first
    le = LabelEncoder()
    categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 
                       'Outlet_Type', 'Item_Type_Combined', 'Item_MRP_Clusters', 'Item_Category', 
                       'Item_Type_New', 'Outlet_Age_Group', 'Price_Range']
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Create interaction features after encoding
    df['Item_Outlet_Interaction'] = df['Item_Identifier'] + '_' + df['Outlet_Identifier']
    df['Item_Type_Outlet_Type'] = df['Item_Type'].astype(str) + '_' + df['Outlet_Type'].astype(str)
    df['Item_Fat_Outlet_Type'] = df['Item_Fat_Content'].astype(str) + '_' + df['Outlet_Type'].astype(str)
    df['Item_MRP_Outlet_Type'] = df['Item_MRP_Clusters'].astype(str) + '_' + df['Outlet_Type'].astype(str)
    df['Item_Category_Outlet_Type'] = df['Item_Category'].astype(str) + '_' + df['Outlet_Type'].astype(str)
    df['Item_Type_New_Outlet_Type'] = df['Item_Type_New'].astype(str) + '_' + df['Outlet_Type'].astype(str)
    df['Outlet_Age_Group_Outlet_Type'] = df['Outlet_Age_Group'].astype(str) + '_' + df['Outlet_Type'].astype(str)
    df['Price_Range_Outlet_Type'] = df['Price_Range'].astype(str) + '_' + df['Outlet_Type'].astype(str)
    
    # Encode interaction features
    interaction_cols = ['Item_Outlet_Interaction', 'Item_Type_Outlet_Type', 'Item_Fat_Outlet_Type',
                       'Item_MRP_Outlet_Type', 'Item_Category_Outlet_Type', 'Item_Type_New_Outlet_Type',
                       'Outlet_Age_Group_Outlet_Type', 'Price_Range_Outlet_Type']
    
    for col in interaction_cols:
        df[col] = le.fit_transform(df[col])
    
    # Target encoding for categorical variables
    target_cols = ['Item_Type', 'Outlet_Type', 'Item_Fat_Content', 'Item_Type_Combined',
                   'Item_MRP_Clusters', 'Item_Category', 'Item_Type_New', 'Outlet_Age_Group',
                   'Price_Range']
    
    if is_train:
        # Store target encoding mappings during training
        for col in target_cols:
            target_mean = df.groupby(col)['Item_Outlet_Sales'].mean()
            target_encoding_maps[col] = target_mean
            df[f'{col}_Mean_Sales'] = df[col].map(target_mean)
    else:
        # Apply stored target encoding mappings during testing
        for col in target_cols:
            df[f'{col}_Mean_Sales'] = df[col].map(target_encoding_maps[col])
            # Fill missing values with global mean
            global_mean = target_encoding_maps[col].mean()
            df[f'{col}_Mean_Sales'] = df[f'{col}_Mean_Sales'].fillna(global_mean)
    
    # Scale numerical features
    numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Store or check feature columns
    if is_train:
        train_feature_columns = df.columns.difference(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
    else:
        # Ensure test data has all training features
        missing_cols = set(train_feature_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        # Ensure columns are in the same order
        df = df[train_feature_columns]
    
    return df

train = preprocess_data(train)
test = preprocess_data(test, is_train=False)
print("✓ Data preprocessing completed!")

# Prepare training data
print_progress(3, 7, "Preparing training data...", time.time())
X = train.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)
y = train['Item_Outlet_Sales']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("✓ Training data prepared!")

# Define models with optimized hyperparameters
print_progress(4, 7, "Training and tuning individual models...", time.time())
base_models = {
    'CatBoost': {
        'model': CatBoostRegressor(random_state=42, verbose=0),
        'params': {
            'iterations': [300],
            'learning_rate': [0.05],
            'depth': [6],
            'l2_leaf_reg': [3]
        }
    },
    'LightGBM': {
        'model': LGBMRegressor(random_state=42, verbose=-1),
        'params': {
            'n_estimators': [300],
            'learning_rate': [0.05],
            'max_depth': [6],
            'num_leaves': [31]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [300],
            'learning_rate': [0.05],
            'max_depth': [6]
        }
    }
}

# Train and evaluate models
model_results = {}
for i, (name, model_info) in enumerate(base_models.items(), 1):
    print(f"\nModel {i}/{len(base_models)}: Tuning {name}...")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    val_predictions = best_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    
    model_results[name] = {
        'model': best_model,
        'rmse': rmse,
        'params': grid_search.best_params_,
        'predictions': val_predictions
    }
    
    elapsed = time.time() - start_time
    print(f"✓ {name} tuned successfully!")
    print(f"RMSE: {rmse:.2f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Time taken: {elapsed:.2f} seconds")

# Calculate optimal weights based on inverse RMSE
print_progress(5, 7, "Creating ensemble model...", time.time())
rmse_values = np.array([result['rmse'] for result in model_results.values()])
inverse_rmse = 1 / rmse_values
weights = inverse_rmse / np.sum(inverse_rmse)

# Weighted Voting Regressor
voting_regressor = VotingRegressor(
    estimators=[(name, result['model']) for name, result in model_results.items()],
    weights=weights
)
voting_regressor.fit(X_train, y_train)
voting_predictions = voting_regressor.predict(X_val)
voting_rmse = np.sqrt(mean_squared_error(y_val, voting_predictions))
model_results['Weighted Voting Regressor'] = {
    'model': voting_regressor,
    'rmse': voting_rmse,
    'params': {'weights': dict(zip(model_results.keys(), weights))}
}

print("✓ Ensemble model created successfully!")

# Select best model
print_progress(6, 7, "Selecting best model...", time.time())
best_model_name = min(model_results.items(), key=lambda x: x[1]['rmse'])[0]
best_model = model_results[best_model_name]['model']
best_rmse = model_results[best_model_name]['rmse']
print(f"Best model: {best_model_name} with RMSE: {best_rmse:.2f}")

# Make predictions
print_progress(7, 7, "Making predictions...", time.time())
test_features = test.copy()
for col in ['Item_Identifier', 'Outlet_Identifier']:
    if col in test_features.columns:
        test_features = test_features.drop(col, axis=1)
test_predictions = best_model.predict(test_features)
test_predictions = np.maximum(test_predictions, 0)  # Ensure no negative predictions

# Create submission file
submission['Item_Outlet_Sales'] = test_predictions
submission.to_csv('final_submission_enhanced.csv', index=False)
print("✓ Enhanced submission file created successfully!")

# Save results to markdown file
with open('model_results.md', 'w') as f:
    f.write("# Model Performance Results\n\n")
    
    # Model Performance Comparison
    f.write("## Model Performance Comparison\n\n")
    f.write("| Model | RMSE | Best Parameters |\n")
    f.write("|-------|------|-----------------|\n")
    for name, result in model_results.items():
        f.write(f"| {name} | {result['rmse']:.2f} | {result['params']} |\n")
    
    # Best Model Details
    f.write("\n## Best Model Details\n\n")
    f.write(f"- Best Model: {best_model_name}\n")
    f.write(f"- RMSE: {best_rmse:.2f}\n")
    f.write(f"- Parameters: {model_results[best_model_name]['params']}\n")
    
    # Feature Importance Analysis
    f.write("\n## Feature Importance Analysis\n\n")
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        f.write("Top 10 Most Important Features:\n\n")
        f.write("| Feature | Importance |\n")
        f.write("|---------|------------|\n")
        for _, row in feature_importance.head(10).iterrows():
            f.write(f"| {row['Feature']} | {row['Importance']:.4f} |\n")
    else:
        f.write("Feature importance not available for the selected model.\n")
    
    # Data Preprocessing Summary
    f.write("\n## Data Preprocessing Summary\n\n")
    f.write("1. Missing Value Handling:\n")
    f.write("   - Item_Weight: Filled with mean\n")
    f.write("   - Outlet_Size: Filled with 'Medium'\n\n")
    
    f.write("2. Feature Engineering:\n")
    f.write("   - Created Item_Type_Combined from Item_Identifier\n")
    f.write("   - Created Item_MRP_Clusters using quantiles\n")
    f.write("   - Created Item_Category based on Item_Type\n")
    f.write("   - Created Item_Type_New for perishable/non-perishable items\n")
    f.write("   - Created Outlet_Age and Outlet_Age_Group\n")
    f.write("   - Created Price_Range using quantiles\n")
    f.write("   - Created various interaction features\n\n")
    
    f.write("3. Feature Encoding:\n")
    f.write("   - Label encoded all categorical variables\n")
    f.write("   - Applied target encoding for categorical variables\n\n")
    
    f.write("4. Feature Scaling:\n")
    f.write("   - Scaled numerical features using StandardScaler\n")
    
    # Model Tuning Summary
    f.write("\n## Model Tuning Summary\n\n")
    f.write("1. Base Models:\n")
    for name, result in model_results.items():
        if name not in ['Weighted Voting Regressor']:
            f.write(f"   - {name}: RMSE = {result['rmse']:.2f}\n")
    
    f.write("\n2. Ensemble Models:\n")
    f.write(f"   - Weighted Voting Regressor: RMSE = {model_results['Weighted Voting Regressor']['rmse']:.2f}\n")

print("✓ Results saved to model_results.md") 