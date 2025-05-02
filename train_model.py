import logging
logging.basicConfig(filename='train_model.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Started training process.')

# Step 1: Load the dataset
import numpy as np 
import pandas as pd 
import os

# Log the dataset loading process
logging.info('Loading dataset...')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        logging.info(f'Found file: {os.path.join(dirname, filename)}')

df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')
logging.info(f'Dataset loaded with shape: {df.shape}')

# Display basic information
logging.info('Displaying dataset info:')
df.info()

# Step 2: Visualizing the Data
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Plotting the distribution of the target column (Median House Value)
logging.info('Plotting distribution of median_house_value.')
plt.figure(figsize=(8, 5))
sns.histplot(df['median_house_value'], kde=True, bins=30)
plt.title("Distribution of house prices")
plt.xlabel("Price (in $)")
plt.ylabel("Frequency")
plt.show()

# Step 3: Correlation Heatmap for Numeric Features
logging.info('Plotting correlation heatmap for numeric features.')
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Preparing Features and Target
logging.info('Preparing features (X) and target (y).')
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Step 5: Train-Test Split (80-20 Split)
from sklearn.model_selection import train_test_split
logging.info('Performing train-test split (80-20).')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logging.info(f"Training Samples: {X_train.shape[0]}")
logging.info(f"Testing Samples: {X_test.shape[0]}")

# Step 6: Feature Scaling and One-Hot Encoding
logging.info('Starting feature scaling and one-hot encoding for categorical columns.')
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify categorical and numerical columns
categorical_cols = ['ocean_proximity']
numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Step 7: Fit and Transform Features
logging.info('Fitting and transforming training data.')
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Step 8: Try Multiple Models
logging.info('Starting model training and comparison (5-fold cross-validation).')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# Model comparison via RMSE
for name, model in models.items():
    logging.info(f"Training {name} model.")
    scores = cross_val_score(model, X_train_processed, y_train, cv=5, scoring='neg_root_mean_squared_error')
    rmse_scores = -scores
    logging.info(f"{name}: RMSE = {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")

# Step 9: Hyperparameter Tuning for XGBoost
logging.info('Starting hyperparameter tuning for XGBoost using GridSearchCV.')
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the grid search to the training data
grid.fit(X_train_processed, y_train)
logging.info(f"Best XGBoost Parameters: {grid.best_params_}")

# Step 10: Evaluate the Final Model
logging.info('Evaluating final model on test set.')

final_model = grid.best_estimator_
final_model.fit(X_train_processed, y_train)

from sklearn.metrics import mean_squared_error, r2_score

y_pred = final_model.predict(X_test_processed)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

logging.info(f"Final Model Performance on Test Set:")
logging.info(f"Test RMSE: {rmse:.3f}")
logging.info(f"Test R² Score: {r2:.3f}")

# Step 11: Save the Model and Scaler
import joblib

logging.info('Saving the trained model and scaler.')
joblib.dump(final_model, 'model.pkl')
joblib.dump(preprocessor, 'scaler.pkl')

logging.info('Model and scaler saved successfully as model.pkl and scaler.pkl')

logging.info('Training process completed successfully.')
