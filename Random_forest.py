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
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

warnings.filterwarnings("ignore")

# Prepare to capture print statements
output = StringIO()

# Load the CSV file from your local path
file_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\CarStory (2).csv'
df = pd.read_csv(file_path)

# Basic data inspection
output.write("Basic Data Inspection\n")
output.write("="*30 + "\n")
output.write(str(df.head()) + "\n\n")
output.write(str(df.info()) + "\n\n")
output.write(str(df.describe()) + "\n\n")

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
output.write("Missing Data Check\n")
output.write("="*30 + "\n")
if df.isnull().sum().sum() == 0:
    output.write("No missing data.\n\n")
else:
    output.write("Missing values in each column:\n")
    output.write(str(df.isnull().sum()) + "\n\n")

# Save processed data without 'price' normalization
output_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\Processed_CarData.csv'
df.to_csv(output_path, index=False)

# Normalize specified columns (excluding 'price')
numeric_cols = ['mileage', 'tax', 'mpg', 'engineSize']
scaler = MinMaxScaler(feature_range=(1, 10))
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Feature engineering for clustering with a tax-centric approach
df['tax_per_engine_size'] = df['tax'] / (df['engineSize'] + 1)  # Avoid division by zero
df['tax_per_mileage'] = df['tax'] / (df['mileage'] + 1)

# Clustering with KMeans using tax-centric features
features_for_clustering = df[['tax', 'tax_per_engine_size', 'tax_per_mileage']].fillna(df[['tax', 'tax_per_engine_size', 'tax_per_mileage']].median())
features_scaled = StandardScaler().fit_transform(features_for_clustering)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans_labels = kmeans.fit_predict(features_scaled)
df['kmeans_cluster'] = kmeans_labels

# Calculate silhouette score to evaluate clustering quality
silhouette = silhouette_score(features_scaled, kmeans_labels)
output.write("K-Means Clustering with Tax-Centric Features\n")
output.write("="*50 + "\n")
output.write(f"K-Means Silhouette Score: {silhouette}\n\n")

# Plot tax-centric clustering
plt.figure(figsize=(10, 6))
plt.scatter(df['tax_per_engine_size'], df['tax_per_mileage'], c=df['kmeans_cluster'], cmap='viridis', marker='o', s=50, alpha=0.7)
plt.xlabel('Tax per Engine Size')
plt.ylabel('Tax per Mileage')
plt.title('K-Means Clustering of Cars by Tax-Centric Features')
plt.colorbar(label='Cluster')
cluster_plot_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\cluster_tax_plot.png'
plt.savefig(cluster_plot_path)  # Save the plot for the PDF
plt.close()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Set up data for training with the tax-centric features
X = df[['mileage', 'tax', 'mpg', 'engineSize', 'tax_per_engine_size', 'tax_per_mileage']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = linear_model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

output.write("Linear Regression Model Performance for Price Prediction\n")
output.write("="*50 + "\n")
output.write(f"R-squared Score: {r2}\n")
output.write(f"Mean Absolute Error (MAE): {mae}\n")
output.write(f"Mean Squared Error (MSE): {mse}\n\n")

# Display metrics
print("***********************************************************")
print("Linear Regression Model Performance for Price Prediction:")
print(f"R-squared Score: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print("***********************************************************")


# Set up data for training
X = df[['mileage', 'tax', 'mpg', 'engineSize', 'tax_per_engine_size', 'tax_per_mileage']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5],
    'min_samples_split': [3],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

# Initialize and run Grid Search
rf = RandomForestRegressor(random_state=0)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

output.write("Grid Search Results\n")
output.write("="*30 + "\n")
output.write(f"Best Parameters: {best_params}\n")
output.write(f"Best R-squared Score from Grid Search: {best_score}\n\n")

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

output.write("Model Evaluation\n")
output.write("="*30 + "\n")
output.write(f"R-squared Score: {r2}\n")
output.write(f"Mean Absolute Error (MAE): {mae}\n")
output.write(f"Mean Squared Error (MSE): {mse}\n\n")

# Save the best model locally
model_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\random_forest_price_model.joblib'
joblib.dump(best_rf, model_path)

# Generate PDF with results
pdf_path = r'C:\\Users\\zaian\\OneDrive\\Desktop\\CARtest\\output_report.pdf'
pdf = canvas.Canvas(pdf_path, pagesize=letter)

# Write text to PDF with spacing between sections
pdf.setFont("Helvetica", 10)
pdf.drawString(40, 750, "Car Data Analysis Report")
y = 730
for line in output.getvalue().splitlines():
    if y < 100:
        pdf.showPage()
        y = 750
        pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, line)
    y -= 12

# Add clustering plot to PDF
pdf.showPage()  # Start a new page for the image
pdf.drawImage(cluster_plot_path, 100, 400, width=400, height=300)
pdf.setFont("Helvetica", 12)
pdf.drawString(40, 750, "K-Means Clustering Plot")
pdf.save()

print(f"PDF report saved at '{pdf_path}'")
