import os
import joblib
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score
)

warnings.filterwarnings("ignore")

# Load the CSV file from your local path
file_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\CarStory (2).csv'
df = pd.read_csv(file_path)

# Basic data inspection
print(df.head())
print(df.info())
print(df.describe())

# Handle NaNs in 'tax' and drop unnecessary columns
df['tax'] = df['tax'].fillna(df['tax(£)'])
df.drop(columns=['tax(£)', 'Unnamed: 0', 'model'], inplace=True)

# Fill missing values in 'tax' based on the median of 'engineSize' and 'year'
median_tax_by_group = df.groupby(['engineSize', 'year'])['tax'].median()
df['tax'] = df.apply(lambda row: median_tax_by_group.get((row['engineSize'], row['year']), df['tax'].median()) if pd.isnull(row['tax']) else row['tax'], axis=1)

# Filter out future years
current_year = datetime.now().year
df = df[df['year'] <= current_year]

# Fill remaining missing values with median values
df['tax'] = df['tax'].fillna(df['tax'].median())
df['mpg'] = df['mpg'].fillna(df['mpg'].median())

# Convert categorical columns to lowercase
df['transmission'] = df['transmission'].str.lower().astype('category')
df['fuelType'] = df['fuelType'].str.lower().astype('category')

# Check for any remaining missing values
if df.isnull().sum().sum() == 0:
    print("***********************************************************")
    print("No missing data")
    print("***********************************************************")
else:
    print("***********************************************************")
    print("Missing values in each column:")
    print(df.isnull().sum())
    print("***********************************************************")

# Save processed data without 'price' normalization
output_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\Processed_CarData.csv'
df.to_csv(output_path, index=False)
print("Processed data saved to 'Processed_CarData.csv'")

# Normalize specified columns (excluding 'price')
numeric_cols = ['mileage', 'tax', 'mpg', 'engineSize']
scaler = MinMaxScaler(feature_range=(1, 10))
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Save normalized data locally as CSV
output_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\Car_Normalized.csv'
df.to_csv(output_path, index=False)
print("Normalized data saved to 'Car_Normalized.csv'")

# Feature engineering for clustering and regression
df['price_per_mileage'] = df['price'] / (df['mileage'] + 1)
df['tax_to_price_ratio'] = df['tax'] / (df['price'] + 1)

# Clustering with KMeans
features_for_clustering = df[['tax_to_price_ratio', 'price_per_mileage']].fillna(df[['tax_to_price_ratio', 'price_per_mileage']].median())
features_scaled = StandardScaler().fit_transform(features_for_clustering)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(features_scaled)
df['kmeans_cluster'] = kmeans_labels
print("***********************************************************")
print(f"K-Means Silhouette Score: {silhouette_score(features_scaled, kmeans_labels)}")
print("***********************************************************")

plt.figure(figsize=(10, 6))
plt.scatter(df['tax_to_price_ratio'], df['price_per_mileage'], c=df['kmeans_cluster'], cmap='viridis', marker='o', s=50, alpha=0.7)
plt.xlabel('Tax to Price Ratio')
plt.ylabel('Price per Mileage')
plt.title('K-Means Clustering of Cars by Cost-Efficiency Features')
plt.colorbar(label='Cluster')
plt.show()

# Set up data for training
X = df[['mileage', 'tax', 'mpg', 'engineSize', 'price_per_mileage', 'tax_to_price_ratio']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200],   # Two choices for number of trees
    'max_depth': [5],             # Fixed to 5 to limit depth
    'min_samples_split': [3],  # Two choices for minimum samples required to split
    'min_samples_leaf': [1, 2],   # Two choices for minimum samples at leaf
    'max_features': ['sqrt']      # Only one option for max features
}


# Initialize and run Grid Search
rf = RandomForestRegressor(random_state=0)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best parameters and score from Grid Search
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("***********************************************************")
print("Best Parameters:", best_params)
print("Best R-squared Score from Grid Search:", best_score)
print("***********************************************************")

# Train final model with the best parameters
best_rf = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    random_state=0
)
best_rf.fit(X_train, y_train)

# Predict and evaluate the tuned model
y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("***********************************************************")
print("Tuned Random Forest Regression Performance for Price Prediction:")
print(f"R-squared Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print("***********************************************************")

# Save the best model locally
model_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\random_forest_price_model.joblib'
joblib.dump(best_rf, model_path)
print(f"Model saved at '{model_path}'")
