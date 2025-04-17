import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    print("Loading data...")
    # Load the datasets
    train = pd.read_csv('train_v9rqX0R.csv')
    test = pd.read_csv('test_AbJTz2l.csv')
    sample_submission = pd.read_csv('sample_submission_8RXa3c6.csv')
    print("Data loaded successfully!")

    print("\nPreprocessing data...")
    # Combine train and test for preprocessing
    train['source'] = 'train'
    test['source'] = 'test'
    data = pd.concat([train, test], ignore_index=True)

    # Store ID columns for later use
    id_columns = ['Item_Identifier', 'Outlet_Identifier']

    # Data Preprocessing
    def preprocess_data(df):
        # Handle missing values
        df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
        df['Outlet_Size'].fillna('Medium', inplace=True)
        
        # Feature Engineering
        # Create a new feature 'Item_Type_Combined' by combining similar item types
        df['Item_Type_Combined'] = df['Item_Type'].apply(lambda x: x[0:2])
        
        # Create a new feature 'Years_Established'
        df['Years_Established'] = 2013 - df['Outlet_Establishment_Year']
        
        # Create a new feature 'Item_MRP_Clusters'
        df['Item_MRP_Clusters'] = pd.qcut(df['Item_MRP'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Label Encoding for categorical variables
        le = LabelEncoder()
        categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                           'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Combined',
                           'Item_MRP_Clusters']
        
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
        
        # Drop unnecessary columns
        df.drop(['Outlet_Establishment_Year'] + id_columns, axis=1, inplace=True)
        
        return df

    # Preprocess the data
    data = preprocess_data(data.copy())
    print("Data preprocessing completed!")

    print("\nPreparing training data...")
    # Split back into train and test
    train_processed = data[data['source'] == 'train'].drop('source', axis=1)
    test_processed = data[data['source'] == 'test'].drop(['source', 'Item_Outlet_Sales'], axis=1)

    # Prepare training data
    X = train_processed.drop('Item_Outlet_Sales', axis=1)
    y = train_processed['Item_Outlet_Sales']

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training data prepared!")

    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Train and evaluate models
    best_model = None
    best_rmse = float('inf')
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        results[name] = rmse
        print(f"{name} RMSE: {rmse}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    print("\nModel Performance Summary:")
    for name, rmse in results.items():
        print(f"{name}: RMSE = {rmse}")

    print(f"\nTraining best model ({best_model.__class__.__name__}) on full training data...")
    best_model.fit(X, y)

    print("\nMaking predictions on test data...")
    # Make predictions on test data
    test_predictions = best_model.predict(test_processed)

    # Create submission file
    submission = pd.DataFrame({
        'Item_Identifier': test['Item_Identifier'],
        'Outlet_Identifier': test['Outlet_Identifier'],
        'Item_Outlet_Sales': test_predictions
    })

    # Save submission file
    submission.to_csv('final_submission.csv', index=False)
    print("\nSubmission file created successfully!")

except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
    print("Error details:", e.__class__.__name__)
    import traceback
    print("\nFull error traceback:")
    print(traceback.format_exc()) 